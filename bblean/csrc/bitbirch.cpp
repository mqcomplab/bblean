// BitBirch tree, the C++ version of the python tree of bblean.BitBirch (which
// uses it when the extension is available, for the builtin merge criteria). It
// builds exactly the same tree.
//
// - Insertion: the closest subcluster of a node is the first maximum of the
//   Tanimoto similarity to the centroids of its subclusters. At a leaf the
//   nominee is merged with the closest subcluster if the merge criterion
//   accepts it, otherwise it is appended, and the node is split if it has more
//   than branching_factor subclusters. Subclusters on the path ("tracking"
//   subclusters) are updated after the insertion below them.
// - Splits: the most dissimilar pair of centroids of the node (kernels.hpp)
//   seeds the two new nodes, every subcluster goes to the node of the seed it
//   is more similar to (ties to the second one), the first seed to the first.
// - Centroids are majority votes of the linear sums (kernels.hpp).
// - Merge criteria: the builtin criteria of bblean.merges (accept_merge).
//
// Storage: each node keeps the centroids of its subclusters contiguously (in
// order) and their popcounts. Each subcluster ("entry") keeps n_samples, the
// sums sum_kq = sum(ls) and sum_kqsq = sum(ls^2) of its linear sum ls, and ls
// itself, stored
// with the smallest unsigned type that holds n_samples. Entries formed by a
// single fingerprint (possibly weighted) store no linear sum: it is
// n_samples * fingerprint. Molecule indices are kept as linked lists (only for
// leaf entries), so merging clusters is O(1).
//
// Fingerprints are rows of 64 bit words (bit p is feature p ^ 7 of bblean's
// packed representation, see kernels.hpp). Linear sums are kept in this bit
// order, and converted to feature order on import / export.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels.hpp"

namespace py = pybind11;

namespace bblean {

// ------------------------------------------------------------ merge criteria

enum MergeKind : int {
    kDiameter = 0,
    kRadius = 1,
    kToleranceDiameter = 2,
    kFlexibleToleranceDiameter = 3,
    kToleranceRadius = 4,
    kToleranceLegacy = 5,
    kNever = 6,
};

// Parameters of a builtin merge criterion (set by the python BitBirch)
struct MergeCriterion {
    int kind;
    double threshold;
    double tolerance;
    double decay;  // size dependent tolerance (tolerance-* except legacy)
    double offset;

    double tolerance_for(uint64_t old_n) const {
        const double t =
            tolerance *
            (std::exp(-decay * static_cast<double>(old_n)) - offset);
        return (0.0 > t) ? 0.0 : t;
    }
};

// Complement of the iSIM radius of n objects (n >= 2) with column sums ls,
// given sum_kq, sum_kqsq, the popcount cent_pop of their centroid and cent_sum
// = sum_{j in centroid} ls_j. Adding the centroid to the objects gives sums
// sum_kq + cent_pop and sum_kqsq + 2 cent_sum + cent_pop.
inline double radius_compl(uint64_t sum_kq, uint64_t sum_kqsq, uint64_t n,
                           uint64_t cent_pop, uint64_t cent_sum) {
    const double jt = isim(sum_kq, sum_kqsq, static_cast<int64_t>(n));
    const double jt1 =
        isim(sum_kq + cent_pop, sum_kqsq + 2 * cent_sum + cent_pop,
             static_cast<int64_t>(n + 1));
    return (jt1 * static_cast<double>(n + 1) -
            jt * static_cast<double>(n - 1)) /
           2;
}

// Sizes and sums of the old cluster, the nominee and the merged cluster. The
// radius complements are only computed when the criterion needs them
// (new_rc() of the merged cluster, old_rc() of the old one).
struct MergeSums {
    uint64_t old_n, nom_n, new_n;
    uint64_t old_sum_kq, old_sum_kqsq, new_sum_kq, new_sum_kqsq;
};

// Same as check_merge of the builtin criteria of bblean.merges
template <typename NewRc, typename OldRc>
inline bool accept_merge(const MergeCriterion& c, const MergeSums& s,
                         NewRc&& new_rc, OldRc&& old_rc) {
    const double thr = c.threshold;
    switch (c.kind) {
        case kDiameter:
            return isim(s.new_sum_kq, s.new_sum_kqsq,
                        static_cast<int64_t>(s.new_n)) >= thr;
        case kRadius:
            return new_rc() >= thr;
        case kToleranceDiameter:
        case kFlexibleToleranceDiameter: {
            const double new_dc = isim(s.new_sum_kq, s.new_sum_kqsq,
                                       static_cast<int64_t>(s.new_n));
            if (new_dc < thr) {
                return false;
            }
            if (s.old_n == 1) {
                return true;
            }
            const double old_dc = isim(s.old_sum_kq, s.old_sum_kqsq,
                                       static_cast<int64_t>(s.old_n));
            double ref = old_dc;
            if (c.kind == kFlexibleToleranceDiameter && thr < old_dc) {
                ref = thr;  // min(old_dc, thr)
            }
            return new_dc >= ref - c.tolerance_for(s.old_n);
        }
        case kToleranceRadius: {
            const double new_rc_value = new_rc();
            if (new_rc_value < thr) {
                return false;
            }
            if (s.old_n == 1) {
                return true;
            }
            return new_rc_value >= old_rc() - c.tolerance_for(s.old_n);
        }
        case kToleranceLegacy: {
            const double new_dc = isim(s.new_sum_kq, s.new_sum_kqsq,
                                       static_cast<int64_t>(s.new_n));
            if (new_dc < thr) {
                return false;
            }
            if (s.old_n == 1 || s.nom_n != 1) {
                return true;
            }
            const double old_dc = isim(s.old_sum_kq, s.old_sum_kqsq,
                                       static_cast<int64_t>(s.old_n));
            return (new_dc * static_cast<double>(s.new_n) -
                    old_dc * static_cast<double>(s.old_n - 1)) /
                       2 >=
                   old_dc - c.tolerance;
        }
        default:
            return false;
    }
}

// -------------------------------------------------------------------- tree

static inline int width_for(uint64_t n) {
    if (n <= 0xFFull) {
        return 1;
    }
    if (n <= 0xFFFFull) {
        return 2;
    }
    if (n <= 0xFFFFFFFFull) {
        return 4;
    }
    return 8;
}

template <typename F>
static inline decltype(auto) with_sum(uint8_t width, uint8_t* data, F&& f) {
    switch (width) {
        case 1:
            return f(data);
        case 2:
            return f(reinterpret_cast<uint16_t*>(data));
        case 4:
            return f(reinterpret_cast<uint32_t*>(data));
        default:
            return f(reinterpret_cast<uint64_t*>(data));
    }
}

// count values as a numpy array of the smallest uint dtype that holds n
static py::array uint_array(const uint64_t* values, size_t count, uint64_t n) {
    auto make = [&](auto tag) -> py::array {
        using V = decltype(tag);
        py::array_t<V> a(static_cast<py::ssize_t>(count));
        auto* p = a.mutable_data();
        for (size_t j = 0; j < count; ++j) {
            p[j] = static_cast<V>(values[j]);
        }
        return std::move(a);
    };
    switch (width_for(n)) {
        case 1:
            return make(uint8_t{});
        case 2:
            return make(uint16_t{});
        case 4:
            return make(uint32_t{});
        default:
            return make(uint64_t{});
    }
}

struct Entry {
    uint64_t n_samples = 0;
    uint64_t sum_kq = 0;    // sum of the linear sum
    uint64_t sum_kqsq = 0;  // sum of squares of the linear sum
    int32_t child = -1;
    // Bytes per linear sum element; 0: implicit (n_samples * centroid)
    uint8_t width = 0;
    std::vector<uint8_t> sum;
    int64_t head = -1, tail = -1;  // molecule index chain (leaf entries only)
};

struct Node {
    int bf = 0;
    bool leaf = false;
    int32_t prev = -1, next = -1;  // leaf chain, prev == -1 means "first leaf"
    std::vector<int32_t> ents;
    std::vector<uint64_t> cent;  // (bf + 1) * w64
    std::vector<uint32_t> pop;   // (bf + 1)
};

// A subcluster about to be inserted: either one (weighted) fingerprint, whose
// linear sum is n * fp, or a dense linear sum (uint64, internal bit order).
struct Nominee {
    // Centroid (the fingerprint itself if single)
    const uint64_t* cent = nullptr;
    uint32_t pop = 0;
    uint64_t n_samples = 0, sum_kq = 0, sum_kqsq = 0;
    bool dense = false;
    const uint64_t* dense_ls = nullptr;
    int64_t head = -1, tail = -1;
};

class Tree {
   public:
    explicit Tree(int n_features) {
        if (n_features <= 0 || n_features % 8 != 0) {
            throw std::invalid_argument(
                "n_features must be a positive multiple of 8");
        }
        n_features_ = n_features;
        n_bytes_ = n_features / 8;
        w64_ = (n_bytes_ + 7) / 8;
        n_bits_ = w64_ * 64;
        tmp_a_.assign(n_bits_, 0);
        tmp_b_.assign(n_bits_, 0);
        tmp_cent_.assign(w64_, 0);
        row_buf_.assign(w64_, 0);
        tmp_out_.assign(n_bits_, 0);
    }

    int n_features() const { return n_features_; }
    int n_bytes() const { return n_bytes_; }
    bool initialized() const { return root_ >= 0 || first_leaf_ >= 0; }
    bool only_leaves() const { return root_ < 0 && first_leaf_ >= 0; }

    // ------------------------------------------------------------- fitting

    // Fit packed fingerprints (rows of n_features / 8 bytes). Molecule indices
    // are ids[i] if given, else start + i; weights (optional) per row.
    void fit_packed(const uint8_t* X, int64_t rows, const int64_t* ids,
                    int64_t start, const uint64_t* weights,
                    const MergeCriterion& merge, int bf) {
        prepare(merge, bf);
        for (int64_t i = 0; i < rows; ++i) {
            uint64_t* fp = row_buf_.data();
            fp[w64_ - 1] = 0;
            std::memcpy(fp, X + i * n_bytes_, n_bytes_);
            const uint64_t w = weights ? weights[i] : 1;
            Nominee nominee;
            nominee.cent = fp;
            nominee.pop = popcount(fp, w64_);
            nominee.n_samples = w;
            nominee.sum_kq = w * nominee.pop;
            nominee.sum_kqsq = w * w * nominee.pop;
            nominee.head = nominee.tail = new_member(ids ? ids[i] : start + i);
            insert_root(nominee, bf);
        }
    }

    // Fit buffers: rows of n_features + 1 unsigned values (linear sum in
    // feature order, then n_samples). Molecule indices: idx[offs[i]:offs[i+1]]
    // (offs may be null: no indices).
    template <typename T>
    void fit_buffers(const T* X, int64_t rows, const int64_t* idx,
                     const int64_t* offs, const MergeCriterion& merge, int bf) {
        prepare(merge, bf);
        std::vector<uint64_t> dense_ls(n_bits_, 0);
        std::vector<uint64_t> cent(w64_, 0);
        for (int64_t i = 0; i < rows; ++i) {
            const T* buf = X + i * (n_features_ + 1);
            for (int p = 0; p < n_features_; ++p) {
                dense_ls[p] = static_cast<uint64_t>(buf[p ^ 7]);
            }
            Nominee nominee;
            nominee.n_samples = static_cast<uint64_t>(buf[n_features_]);
            summarize(dense_ls.data(), nominee.n_samples, cent.data(),
                      nominee.pop, nominee.sum_kq, nominee.sum_kqsq);
            nominee.cent = cent.data();
            nominee.dense = true;
            nominee.dense_ls = dense_ls.data();
            if (offs != nullptr) {
                for (int64_t k = offs[i]; k < offs[i + 1]; ++k) {
                    const int64_t m = new_member(idx[k]);
                    if (nominee.head < 0) {
                        nominee.head = m;
                    } else {
                        mem_next_[nominee.tail] = m;
                    }
                    nominee.tail = m;
                }
            }
            insert_root(nominee, bf);
        }
    }

    // Release internal nodes (BitBirch.delete_internal_nodes)
    void release_internal() {
        if (root_ < 0 || nodes_[root_].leaf) {
            return;
        }
        std::vector<char> is_leaf(nodes_.size(), 0);
        for (int32_t l = first_leaf_; l >= 0; l = nodes_[l].next) {
            is_leaf[l] = 1;
        }
        for (size_t nd = 0; nd < nodes_.size(); ++nd) {
            if (is_leaf[nd]) {
                continue;
            }
            for (const int32_t e : nodes_[nd].ents) {
                free_entry(e);
            }
            Node empty;
            std::swap(nodes_[nd], empty);
        }
        root_ = -1;
    }

    // -------------------------------------------------------------- export

    // Leaf entries in leaf order, or sorted by n_samples (descending, stable)
    std::vector<int32_t> leaf_entries(bool sort) const {
        std::vector<int32_t> out;
        for (int32_t l = first_leaf_; l >= 0; l = nodes_[l].next) {
            out.insert(out.end(), nodes_[l].ents.begin(), nodes_[l].ents.end());
        }
        if (sort) {
            std::stable_sort(out.begin(), out.end(),
                             [this](int32_t a, int32_t b) {
                                 return ents_[a].n_samples > ents_[b].n_samples;
                             });
        }
        return out;
    }

    std::vector<const uint64_t*> leaf_centroids(
        const std::vector<int32_t>& order) const {
        std::vector<const uint64_t*> where(ents_.size(), nullptr);
        for (int32_t l = first_leaf_; l >= 0; l = nodes_[l].next) {
            for (size_t i = 0; i < nodes_[l].ents.size(); ++i) {
                where[nodes_[l].ents[i]] = nodes_[l].cent.data() + i * w64_;
            }
        }
        std::vector<const uint64_t*> out(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            out[i] = where[order[i]];
        }
        return out;
    }

    uint64_t entry_n_samples(int32_t e) const { return ents_[e].n_samples; }
    int64_t member_head(int32_t e) const { return ents_[e].head; }
    int64_t member_next(int64_t m) const { return mem_next_[m]; }
    int64_t member_id(int64_t m) const { return mem_id_[m]; }

    // Linear sum of entry e (centroid cent) in feature order
    void linear_sum(int32_t e, const uint64_t* cent, uint64_t* out) const {
        linear_sum_of(ents_[e], cent, out);
    }

    // Molecule indices of a member chain
    py::list member_list(int64_t head) const {
        py::list out;
        for (int64_t m = head; m >= 0; m = mem_next_[m]) {
            out.append(mem_id_[m]);
        }
        return out;
    }

    // ----------------------------------------------------------- pickling

    std::string serialize() const;
    static Tree deserialize(const std::string& nominee);

   private:
    void prepare(const MergeCriterion& merge, int bf) {
        if (only_leaves()) {
            throw std::runtime_error(
                "Internal nodes were released, call reset() before fit()");
        }
        if (bf < 1) {
            throw std::invalid_argument("branching_factor must be >= 1");
        }
        merge_ = merge;
        if (root_ < 0) {
            root_ = new_node(bf, true);
            first_leaf_ = root_;
        }
    }

    int64_t new_member(int64_t id) {
        mem_id_.push_back(id);
        mem_next_.push_back(-1);
        return static_cast<int64_t>(mem_id_.size()) - 1;
    }

    int32_t new_node(int bf, bool leaf) {
        Node nd;
        nd.bf = bf;
        nd.leaf = leaf;
        nd.ents.reserve(bf + 1);
        nd.cent.assign(static_cast<size_t>(bf + 1) * w64_, 0);
        nd.pop.assign(bf + 1, 0);
        nodes_.push_back(std::move(nd));
        return static_cast<int32_t>(nodes_.size()) - 1;
    }

    int32_t new_entry() {
        if (!free_ents_.empty()) {
            const int32_t e = free_ents_.back();
            free_ents_.pop_back();
            ents_[e] = Entry();
            return e;
        }
        ents_.emplace_back();
        return static_cast<int32_t>(ents_.size()) - 1;
    }

    void free_entry(int32_t e) {
        ents_[e] = Entry();
        free_ents_.push_back(e);
    }

    uint64_t* row(int32_t nid, int idx) {
        return nodes_[nid].cent.data() + static_cast<size_t>(idx) * w64_;
    }

    void set_row(int32_t nid, int idx, const uint64_t* c, uint32_t pop) {
        std::memcpy(row(nid, idx), c, sizeof(uint64_t) * w64_);
        nodes_[nid].pop[idx] = pop;
    }

    void append_entry(int32_t nid, int32_t e, const uint64_t* c, uint32_t pop) {
        Node& nd = nodes_[nid];
        const int idx = static_cast<int>(nd.ents.size());
        if (idx > nd.bf) {
            throw std::runtime_error("node overflow");
        }
        nd.ents.push_back(e);
        set_row(nid, idx, c, pop);
    }

    // f(get_ls) with get_ls(p) = L_p, specialized for the storage of entry
    template <typename F>
    decltype(auto) with_linear_sum(const Entry& entry, const uint64_t* cent,
                                   F&& f) const {
        if (entry.width == 0) {
            const uint64_t n = entry.n_samples;
            return f([cent, n](int p) -> uint64_t {
                return ((cent[p >> 6] >> (p & 63)) & 1) ? n : 0;
            });
        }
        return with_sum(entry.width, const_cast<uint8_t*>(entry.sum.data()),
                        [&](auto* ls) {
                            return f([ls](int p) -> uint64_t {
                                return static_cast<uint64_t>(ls[p]);
                            });
                        });
    }

    void linear_sum_of(const Entry& entry, const uint64_t* cent,
                       uint64_t* out) const {
        dense_sum(entry, cent, tmp_out_.data());
        for (int j = 0; j < n_features_; ++j) {
            out[j] = tmp_out_[j ^ 7];
        }
    }

    // Dense linear sum into out (uint64, internal order)
    void dense_sum(const Entry& entry, const uint64_t* cent,
                   uint64_t* out) const {
        with_linear_sum(entry, cent, [&](auto get_ls) {
            for (int p = 0; p < n_bits_; ++p) {
                out[p] = get_ls(p);
            }
        });
    }

    // Centroid (with its popcount), sum_kq and sum_kqsq of a dense linear sum
    // of n_samples fingerprints
    void summarize(const uint64_t* ls, uint64_t n_samples, uint64_t* cent,
                   uint32_t& pop, uint64_t& sum_kq, uint64_t& sum_kqsq) const {
        std::fill(cent, cent + w64_, 0);
        sum_kq = sum_kqsq = 0;
        for (int p = 0; p < n_bits_; ++p) {
            sum_kq += ls[p];
            sum_kqsq += ls[p] * ls[p];
            if (centroid_bit(ls[p], static_cast<int64_t>(n_samples))) {
                cent[p >> 6] |= uint64_t{1} << (p & 63);
            }
        }
        pop = popcount(cent, w64_);
    }

    // Store a dense linear sum with the width for n_samples n
    void store_sum(Entry& entry, const uint64_t* dense, uint64_t n) {
        entry.width = static_cast<uint8_t>(width_for(n));
        entry.sum.assign(static_cast<size_t>(n_bits_) * entry.width, 0);
        with_sum(entry.width, entry.sum.data(), [&](auto* ls) {
            using V = std::remove_reference_t<decltype(*ls)>;
            for (int p = 0; p < n_bits_; ++p) {
                ls[p] = static_cast<V>(dense[p]);
            }
        });
    }

    // Make sure the entry has an explicit linear sum of width for new_n
    void ensure_width(Entry& entry, const uint64_t* cent, uint64_t new_n) {
        if (entry.width != 0 && entry.width >= width_for(new_n)) {
            return;
        }
        uint64_t* tmp = tmp_b_.data();
        dense_sum(entry, cent, tmp);
        store_sum(entry, tmp, new_n);
    }

    // Index of the closest entry (first maximum of the Tanimoto similarity)
    int closest(int32_t nid, const Nominee& nominee) {
        const Node& nd = nodes_[nid];
        const size_t m = nd.ents.size();
        sims1_.resize(m);
        jt_sims(nd.cent.data(), nd.pop.data(), m, w64_, nominee.cent,
                sims1_.data());
        return static_cast<int>(std::max_element(sims1_.begin(), sims1_.end()) -
                                sims1_.begin());
    }

    // New centroid of entry (centroid old_cent) after adding nominee; also
    // returns cent_sum = sum_{j in new centroid} ls'_j (ls' = merged linear
    // sum), new_sum_kq and new_sum_kqsq.
    void merged_stats(const Entry& entry, const uint64_t* old_cent,
                      const Nominee& nominee, uint64_t* new_cent,
                      uint32_t& new_pop, uint64_t& cent_sum,
                      uint64_t& new_sum_kq, uint64_t& new_sum_kqsq) {
        const uint64_t new_n = entry.n_samples + nominee.n_samples;
        uint64_t cross = 0;
        cent_sum = 0;
        if (!nominee.dense) {
            // ls'_j = ls_j + nominee_n * fp_j; only bits in old_cent | fp can
            // be set in the new centroid
            with_linear_sum(entry, old_cent, [&](auto get_ls) {
                const uint64_t nominee_n = nominee.n_samples;
                for (int k = 0; k < w64_; ++k) {
                    const uint64_t f = nominee.cent[k];
                    uint64_t out = 0;
                    for (uint64_t w = old_cent[k] | f; w; w &= w - 1) {
                        const int b = ctz64(w);
                        const uint64_t old_value = get_ls(k * 64 + b);
                        const uint64_t in_fp = (f >> b) & 1;
                        cross += in_fp * old_value;
                        const uint64_t new_value =
                            old_value + in_fp * nominee_n;
                        if (centroid_bit(new_value, new_n)) {
                            out |= uint64_t{1} << b;
                            cent_sum += new_value;
                        }
                    }
                    new_cent[k] = out;
                }
            });
            cross *= nominee.n_samples;
        } else {
            uint64_t* ls = tmp_a_.data();
            dense_sum(entry, old_cent, ls);
            std::fill(new_cent, new_cent + w64_, 0);
            for (int p = 0; p < n_bits_; ++p) {
                cross += ls[p] * nominee.dense_ls[p];
                const uint64_t new_value = ls[p] + nominee.dense_ls[p];
                if (centroid_bit(new_value, new_n)) {
                    new_cent[p >> 6] |= uint64_t{1} << (p & 63);
                    cent_sum += new_value;
                }
            }
        }
        new_pop = popcount(new_cent, w64_);
        new_sum_kq = entry.sum_kq + nominee.sum_kq;
        new_sum_kqsq = entry.sum_kqsq + nominee.sum_kqsq + 2 * cross;
    }

    bool accept(const Entry& entry, const uint64_t* old_cent, uint32_t old_pop,
                const Nominee& nominee, uint64_t new_sum_kq,
                uint64_t new_sum_kqsq, uint32_t new_pop,
                uint64_t cent_sum) const {
        const MergeSums sums{entry.n_samples,
                             nominee.n_samples,
                             entry.n_samples + nominee.n_samples,
                             entry.sum_kq,
                             entry.sum_kqsq,
                             new_sum_kq,
                             new_sum_kqsq};
        return accept_merge(
            merge_, sums,
            [&] {
                return radius_compl(new_sum_kq, new_sum_kqsq,
                                    entry.n_samples + nominee.n_samples,
                                    new_pop, cent_sum);
            },
            [&] {
                uint64_t old_cent_sum = 0;
                with_linear_sum(entry, old_cent, [&](auto get_ls) {
                    for_each_bit(old_cent, w64_,
                                 [&](int p) { old_cent_sum += get_ls(p); });
                });
                return radius_compl(entry.sum_kq, entry.sum_kqsq,
                                    entry.n_samples, old_pop, old_cent_sum);
            });
    }

    // Add s to entry e at (nid, idx), with precomputed stats
    void absorb(int32_t nid, int idx, const Nominee& nominee,
                const uint64_t* new_cent, uint32_t new_pop, uint64_t new_sum_kq,
                uint64_t new_sum_kqsq, bool members) {
        Entry& entry = ents_[nodes_[nid].ents[idx]];
        const uint64_t new_n = entry.n_samples + nominee.n_samples;
        ensure_width(entry, row(nid, idx), new_n);
        with_sum(entry.width, entry.sum.data(), [&](auto* ls) {
            using V = std::remove_reference_t<decltype(*ls)>;
            if (!nominee.dense) {
                const V add = static_cast<V>(nominee.n_samples);
                for_each_bit(nominee.cent, w64_, [&](int p) {
                    ls[p] = static_cast<V>(ls[p] + add);
                });
            } else {
                for (int p = 0; p < n_bits_; ++p) {
                    ls[p] = static_cast<V>(ls[p] + nominee.dense_ls[p]);
                }
            }
        });
        entry.n_samples = new_n;
        entry.sum_kq = new_sum_kq;
        entry.sum_kqsq = new_sum_kqsq;
        if (members && nominee.head >= 0) {
            if (entry.head < 0) {
                entry.head = nominee.head;
            } else {
                mem_next_[entry.tail] = nominee.head;
            }
            entry.tail = nominee.tail;
        }
        set_row(nid, idx, new_cent, new_pop);
    }

    void append_nominee(int32_t nid, const Nominee& nominee) {
        const int32_t e = new_entry();
        Entry& entry = ents_[e];
        entry.n_samples = nominee.n_samples;
        entry.sum_kq = nominee.sum_kq;
        entry.sum_kqsq = nominee.sum_kqsq;
        entry.head = nominee.head;
        entry.tail = nominee.tail;
        if (nominee.dense) {
            store_sum(entry, nominee.dense_ls, entry.n_samples);
        }
        append_entry(nid, e, nominee.cent, nominee.pop);
    }

    struct SplitOut {
        int32_t e1, e2;
        std::vector<uint64_t> c1, c2;
        uint32_t p1, p2;
    };

    SplitOut split(int32_t nid) {
        const int m = static_cast<int>(nodes_[nid].ents.size());
        const bool leaf = nodes_[nid].leaf;
        const int bf = nodes_[nid].bf;
        const std::vector<int32_t> ents = nodes_[nid].ents;
        const std::vector<uint64_t> rows(
            nodes_[nid].cent.begin(),
            nodes_[nid].cent.begin() + static_cast<size_t>(m) * w64_);
        const std::vector<uint32_t> pops(nodes_[nid].pop.begin(),
                                         nodes_[nid].pop.begin() + m);
        auto row_of = [&](int i) {
            return rows.data() + static_cast<size_t>(i) * w64_;
        };

        sims1_.resize(m);
        sims2_.resize(m);
        const size_t fp1 =
            most_dissimilar(rows.data(), pops.data(), m, w64_, n_bits_,
                            sims1_.data(), sims2_.data())
                .fp1;

        // node1 is new, node2 is the old node (reused)
        const int32_t nid1 = new_node(bf, leaf);
        if (leaf) {
            Node& node2 = nodes_[nid];
            Node& node1 = nodes_[nid1];
            node1.prev = node2.prev;
            if (node2.prev >= 0) {
                nodes_[node2.prev].next = nid1;
            } else {
                first_leaf_ = nid1;
            }
            node1.next = nid;
            node2.prev = nid1;
        }
        nodes_[nid].ents.clear();
        std::vector<uint64_t> acc1(n_bits_, 0), acc2(n_bits_, 0);
        uint64_t n_samples1 = 0, n_samples2 = 0;
        uint64_t* tmp = tmp_a_.data();
        for (int i = 0; i < m; ++i) {
            const bool closer1 =
                static_cast<size_t>(i) == fp1 || sims1_[i] > sims2_[i];
            const Entry& entry = ents_[ents[i]];
            std::vector<uint64_t>& acc = closer1 ? acc1 : acc2;
            dense_sum(entry, row_of(i), tmp);
            for (int p = 0; p < n_bits_; ++p) {
                acc[p] += tmp[p];
            }
            if (closer1) {
                n_samples1 += entry.n_samples;
                append_entry(nid1, ents[i], row_of(i), pops[i]);
            } else {
                n_samples2 += entry.n_samples;
                append_entry(nid, ents[i], row_of(i), pops[i]);
            }
        }
        SplitOut out;
        out.e1 = make_summary(nid1, n_samples1, acc1, out.c1, out.p1);
        out.e2 = make_summary(nid, n_samples2, acc2, out.c2, out.p2);
        return out;
    }

    int32_t make_summary(int32_t child, uint64_t n,
                         const std::vector<uint64_t>& acc,
                         std::vector<uint64_t>& c, uint32_t& pop) {
        const int32_t e = new_entry();
        Entry& entry = ents_[e];
        entry.child = child;
        entry.n_samples = n;
        c.resize(w64_);
        summarize(acc.data(), n, c.data(), pop, entry.sum_kq, entry.sum_kqsq);
        store_sum(entry, acc.data(), n);
        return e;
    }

    // Insert s below node nid; returns whether node nid must be split
    bool insert(int32_t nid, const Nominee& nominee) {
        if (nodes_[nid].ents.empty()) {
            append_nominee(nid, nominee);
            return false;
        }
        const int idx = closest(nid, nominee);
        const int32_t e = nodes_[nid].ents[idx];
        uint64_t* new_cent = tmp_cent_.data();
        uint32_t new_pop;
        uint64_t cent_sum, new_sum_kq, new_sum_kqsq;
        if (ents_[e].child < 0) {
            const uint64_t* old_cent = row(nid, idx);
            merged_stats(ents_[e], old_cent, nominee, new_cent, new_pop,
                         cent_sum, new_sum_kq, new_sum_kqsq);
            if (accept(ents_[e], old_cent, nodes_[nid].pop[idx], nominee,
                       new_sum_kq, new_sum_kqsq, new_pop, cent_sum)) {
                absorb(nid, idx, nominee, new_cent, new_pop, new_sum_kq,
                       new_sum_kqsq, true);
                return false;
            }
            append_nominee(nid, nominee);
            return static_cast<int>(nodes_[nid].ents.size()) > nodes_[nid].bf;
        }
        const int32_t child = ents_[e].child;
        if (insert(child, nominee)) {
            SplitOut o = split(child);
            nodes_[nid].ents[idx] = o.e1;
            set_row(nid, idx, o.c1.data(), o.p1);
            free_entry(e);
            append_entry(nid, o.e2, o.c2.data(), o.p2);
            return static_cast<int>(nodes_[nid].ents.size()) > nodes_[nid].bf;
        }
        // Update the tracking subcluster
        merged_stats(ents_[e], row(nid, idx), nominee, new_cent, new_pop,
                     cent_sum, new_sum_kq, new_sum_kqsq);
        absorb(nid, idx, nominee, new_cent, new_pop, new_sum_kq, new_sum_kqsq,
               false);
        return false;
    }

    void insert_root(const Nominee& nominee, int bf) {
        if (insert(root_, nominee)) {
            SplitOut o = split(root_);
            const int32_t r = new_node(bf, false);
            append_entry(r, o.e1, o.c1.data(), o.p1);
            append_entry(r, o.e2, o.c2.data(), o.p2);
            root_ = r;
        }
    }

    int n_features_ = 0, n_bytes_ = 0, w64_ = 0, n_bits_ = 0;
    MergeCriterion merge_{};
    std::vector<Node> nodes_;
    std::vector<Entry> ents_;
    std::vector<int32_t> free_ents_;
    std::vector<int64_t> mem_id_, mem_next_;
    int32_t root_ = -1, first_leaf_ = -1;
    // scratch
    std::vector<uint64_t> tmp_a_, tmp_b_, tmp_cent_, row_buf_;
    mutable std::vector<uint64_t> tmp_out_;
    std::vector<double> sims1_, sims2_;
};

// ---------------------------------------------------------------- pickling

namespace {
struct Writer {
    std::string out;
    template <typename T>
    void put(const T& v) {
        out.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    template <typename T>
    void vec(const std::vector<T>& v) {
        put<uint64_t>(v.size());
        out.append(reinterpret_cast<const char*>(v.data()),
                   sizeof(T) * v.size());
    }
};
struct Reader {
    const std::string& s;
    size_t pos = 0;
    template <typename T>
    T get() {
        if (pos + sizeof(T) > s.size()) {
            throw std::runtime_error("corrupt BitBirch tree state");
        }
        T v;
        std::memcpy(&v, s.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    template <typename T>
    void vec(std::vector<T>& v) {
        const uint64_t n = get<uint64_t>();
        if (n > (s.size() - pos) / sizeof(T)) {
            throw std::runtime_error("corrupt BitBirch tree state");
        }
        v.resize(n);
        std::memcpy(v.data(), s.data() + pos, sizeof(T) * n);
        pos += sizeof(T) * n;
    }
};
constexpr uint64_t kMagic = 0x3165657274626262ull;  // "bbbtree1"
}  // namespace

std::string Tree::serialize() const {
    Writer w;
    w.put(kMagic);
    w.put<int32_t>(n_features_);
    w.put<int32_t>(root_);
    w.put<int32_t>(first_leaf_);
    w.vec(mem_id_);
    w.vec(mem_next_);
    w.vec(free_ents_);
    w.put<uint64_t>(nodes_.size());
    for (const Node& nd : nodes_) {
        w.put<int32_t>(nd.bf);
        w.put<uint8_t>(nd.leaf);
        w.put<int32_t>(nd.prev);
        w.put<int32_t>(nd.next);
        w.vec(nd.ents);
        w.vec(nd.cent);
        w.vec(nd.pop);
    }
    w.put<uint64_t>(ents_.size());
    for (const Entry& entry : ents_) {
        w.put(entry.n_samples);
        w.put(entry.sum_kq);
        w.put(entry.sum_kqsq);
        w.put(entry.child);
        w.put(entry.width);
        w.put(entry.head);
        w.put(entry.tail);
        w.vec(entry.sum);
    }
    return std::move(w.out);
}

Tree Tree::deserialize(const std::string& s) {
    Reader r{s};
    if (r.get<uint64_t>() != kMagic) {
        throw std::runtime_error("not a BitBirch tree state");
    }
    Tree t(r.get<int32_t>());
    t.root_ = r.get<int32_t>();
    t.first_leaf_ = r.get<int32_t>();
    r.vec(t.mem_id_);
    r.vec(t.mem_next_);
    r.vec(t.free_ents_);
    t.nodes_.resize(r.get<uint64_t>());
    for (Node& nd : t.nodes_) {
        nd.bf = r.get<int32_t>();
        nd.leaf = r.get<uint8_t>() != 0;
        nd.prev = r.get<int32_t>();
        nd.next = r.get<int32_t>();
        r.vec(nd.ents);
        r.vec(nd.cent);
        r.vec(nd.pop);
    }
    t.ents_.resize(r.get<uint64_t>());
    for (Entry& entry : t.ents_) {
        entry.n_samples = r.get<uint64_t>();
        entry.sum_kq = r.get<uint64_t>();
        entry.sum_kqsq = r.get<uint64_t>();
        entry.child = r.get<int32_t>();
        entry.width = r.get<uint8_t>();
        entry.head = r.get<int64_t>();
        entry.tail = r.get<int64_t>();
        r.vec(entry.sum);
    }
    return t;
}

}  // namespace bblean

// ----------------------------------------------------------------- bindings

namespace {

using bblean::MergeCriterion;
using bblean::Tree;

using Int64Array =
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using UInt64Array =
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

// merge: (name, threshold, tolerance, decay, offset) of a builtin criterion of
// bblean.merges (bblean.merges._cpp_merge_params)
MergeCriterion to_criterion(const py::tuple& t) {
    if (t.size() != 5) {
        throw std::invalid_argument("invalid merge parameters");
    }
    static const std::unordered_map<std::string, int> kinds{
        {"diameter", bblean::kDiameter},
        {"radius", bblean::kRadius},
        {"tolerance-diameter", bblean::kToleranceDiameter},
        {"flexible-tolerance-diameter", bblean::kFlexibleToleranceDiameter},
        {"tolerance-radius", bblean::kToleranceRadius},
        {"tolerance-legacy", bblean::kToleranceLegacy},
        {"never", bblean::kNever},
    };
    const auto name = t[0].cast<std::string>();
    const auto kind = kinds.find(name);
    if (kind == kinds.end()) {
        throw std::invalid_argument("unknown merge criterion " + name);
    }
    MergeCriterion c{};
    c.kind = kind->second;
    c.threshold = t[1].cast<double>();
    c.tolerance = t[2].cast<double>();
    c.decay = t[3].cast<double>();
    c.offset = t[4].cast<double>();
    return c;
}

void fit_packed(Tree& tree, const py::array_t<uint8_t, py::array::c_style>& X,
                const py::object& ids, int64_t start, const py::object& weights,
                const py::tuple& merge, int bf) {
    if (X.ndim() != 2) {
        throw std::invalid_argument("fingerprints must be a 2D array");
    }
    if (X.shape(1) != tree.n_bytes()) {
        throw std::invalid_argument(
            "fingerprints have " + std::to_string(X.shape(1)) +
            " bytes, expected " + std::to_string(tree.n_bytes()));
    }
    const int64_t rows = X.shape(0);
    Int64Array id_arr;
    UInt64Array w_arr;
    const int64_t* id_ptr = nullptr;
    const uint64_t* w_ptr = nullptr;
    if (!ids.is_none()) {
        id_arr = ids.cast<Int64Array>();
        if (id_arr.ndim() != 1 || id_arr.shape(0) < rows) {
            throw std::invalid_argument("not enough indices");
        }
        id_ptr = id_arr.data();
    }
    if (!weights.is_none()) {
        w_arr = weights.cast<UInt64Array>();
        if (w_arr.ndim() != 1 || w_arr.shape(0) < rows) {
            throw std::invalid_argument("not enough weights");
        }
        w_ptr = w_arr.data();
    }
    const MergeCriterion m = to_criterion(merge);
    py::gil_scoped_release release;
    tree.fit_packed(X.data(), rows, id_ptr, start, w_ptr, m, bf);
}

void fit_buffers(Tree& tree, const py::array& X, const py::object& idx,
                 const py::object& offs, const py::tuple& merge, int bf) {
    if (X.ndim() != 2 || X.shape(1) != tree.n_features() + 1) {
        throw std::invalid_argument(
            "buffers must have shape (N, n_features + 1)");
    }
    const auto kind = X.dtype().kind();
    if (kind != 'u' && kind != 'i') {
        throw std::invalid_argument("buffers must have an integer dtype");
    }
    const int64_t rows = X.shape(0);
    Int64Array id_arr, off_arr;
    const int64_t* id_ptr = nullptr;
    const int64_t* off_ptr = nullptr;
    if (!offs.is_none()) {
        id_arr = idx.cast<Int64Array>();
        off_arr = offs.cast<Int64Array>();
        if (off_arr.ndim() != 1 || off_arr.shape(0) < rows + 1) {
            throw std::invalid_argument("not enough offsets");
        }
        id_ptr = id_arr.data();
        off_ptr = off_arr.data();
        if (off_ptr[0] < 0 || off_ptr[rows] > id_arr.size()) {
            throw std::invalid_argument("offsets out of bounds");
        }
        for (int64_t i = 0; i < rows; ++i) {
            if (off_ptr[i] > off_ptr[i + 1]) {
                throw std::invalid_argument("offsets must be non-decreasing");
            }
        }
    }
    const MergeCriterion m = to_criterion(merge);
    auto contiguous = py::array::ensure(X, py::array::c_style);
    const void* data = contiguous.data();
    const auto size = X.dtype().itemsize();
    if (size != 1 && size != 2 && size != 4 && size != 8) {
        throw std::invalid_argument("unsupported buffer dtype");
    }
    py::gil_scoped_release release;
    switch (size) {
        case 1:
            return tree.fit_buffers(static_cast<const uint8_t*>(data), rows,
                                    id_ptr, off_ptr, m, bf);
        case 2:
            return tree.fit_buffers(static_cast<const uint16_t*>(data), rows,
                                    id_ptr, off_ptr, m, bf);
        case 4:
            return tree.fit_buffers(static_cast<const uint32_t*>(data), rows,
                                    id_ptr, off_ptr, m, bf);
        default:
            return tree.fit_buffers(static_cast<const uint64_t*>(data), rows,
                                    id_ptr, off_ptr, m, bf);
    }
}

py::list cluster_mol_ids(const Tree& tree, bool sort) {
    const auto order = tree.leaf_entries(sort);
    py::list out(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        out[i] = tree.member_list(tree.member_head(order[i]));
    }
    return out;
}

py::array_t<uint64_t> leaf_sizes(const Tree& tree, bool sort) {
    const auto order = tree.leaf_entries(sort);
    py::array_t<uint64_t> out(static_cast<py::ssize_t>(order.size()));
    auto* p = out.mutable_data();
    for (size_t i = 0; i < order.size(); ++i) {
        p[i] = tree.entry_n_samples(order[i]);
    }
    return out;
}

py::array_t<uint64_t> assignments(const Tree& tree, int64_t n_mols, bool sort) {
    const auto order = tree.leaf_entries(sort);
    py::array_t<uint64_t> out(static_cast<py::ssize_t>(n_mols));
    auto* p = out.mutable_data();
    std::fill(p, p + n_mols, 0);
    for (size_t i = 0; i < order.size(); ++i) {
        for (int64_t m = tree.member_head(order[i]); m >= 0;
             m = tree.member_next(m)) {
            int64_t id = tree.member_id(m);
            if (id < 0) {
                id += n_mols;  // python negative indexing
            }
            if (id < 0 || id >= n_mols) {
                throw py::index_error(
                    "molecule index out of bounds for assignments");
            }
            p[id] = static_cast<uint64_t>(i + 1);
        }
    }
    return out;
}

py::array_t<uint8_t> centroids(const Tree& tree, bool sort) {
    const auto order = tree.leaf_entries(sort);
    const auto where = tree.leaf_centroids(order);
    const int n_bytes = tree.n_bytes();
    py::array_t<uint8_t> out({static_cast<py::ssize_t>(order.size()),
                              static_cast<py::ssize_t>(n_bytes)});
    auto* p = out.mutable_data();
    for (size_t i = 0; i < order.size(); ++i) {
        std::memcpy(p + i * n_bytes, where[i], n_bytes);
    }
    return out;
}

// Linear sum + n_samples buffers of the leaf subclusters, with the smallest
// uint dtype that holds n_samples
py::list leaf_buffers(const Tree& tree, bool sort) {
    const auto order = tree.leaf_entries(sort);
    const auto where = tree.leaf_centroids(order);
    const int F = tree.n_features();
    std::vector<uint64_t> buf(F + 1);
    py::list out(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        tree.linear_sum(order[i], where[i], buf.data());
        buf[F] = tree.entry_n_samples(order[i]);
        out[i] = bblean::uint_array(buf.data(), buf.size(), buf[F]);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_cpp_bitbirch, m) {
    m.doc() = "BitBirch tree (C++ extension)";
    py::class_<Tree>(m, "Tree")
        .def(py::init<int>(), py::arg("n_features"))
        .def_property_readonly("n_features", &Tree::n_features)
        .def_property_readonly("initialized", &Tree::initialized)
        .def_property_readonly("only_leaves", &Tree::only_leaves)
        .def("fit_packed", &fit_packed, py::arg("X"), py::arg("ids"),
             py::arg("start"), py::arg("weights"), py::arg("merge"),
             py::arg("branching_factor"))
        .def("fit_buffers", &fit_buffers, py::arg("X"), py::arg("idx"),
             py::arg("offsets"), py::arg("merge"), py::arg("branching_factor"))
        .def("release_internal", &Tree::release_internal)
        .def("cluster_mol_ids", &cluster_mol_ids, py::arg("sort") = true)
        .def("leaf_sizes", &leaf_sizes, py::arg("sort") = true)
        .def("assignments", &assignments, py::arg("n_mols"),
             py::arg("sort") = true)
        .def("centroids", &centroids, py::arg("sort") = true)
        .def("leaf_buffers", &leaf_buffers, py::arg("sort") = true)
        .def(py::pickle([](const Tree& t) { return py::bytes(t.serialize()); },
                        [](const py::bytes& b) {
                            return Tree::deserialize(std::string(b));
                        }));
}
