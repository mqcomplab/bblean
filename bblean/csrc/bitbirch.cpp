// BitBirch tree. It must build exactly the same tree as the python tree
// (bblean.bitbirch._PyTree).
//
// Fingerprints are rows of 64 bit words, zero padded to whole words, read from
// bblean's packed bytes on a little endian machine: bit p of a row is bit p % 8
// of byte p / 8, which is feature p ^ 7 (np.packbits uses big bit order).
// Linear sums are kept in this bit order, and converted to feature order on
// import / export.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kernels.hpp"

namespace py = pybind11;

namespace bblean {
namespace {

// The tree keeps fingerprints as rows of 64 bit words; the kernels read rows
// as bytes
const uint8_t* bytes(const uint64_t* p) {
    return reinterpret_cast<const uint8_t*>(p);
}

// ------------------------------------------------------------ merge criteria

enum class MergeKind {
    kDiameter,
    kRadius,
    kToleranceDiameter,
    kFlexibleToleranceDiameter,
    kToleranceRadius,
    kToleranceLegacy,
    kNever,
};

// Parameters of a builtin merge criterion (set by the python BitBirch)
struct MergeCriterion {
    MergeKind kind = MergeKind::kNever;
    double threshold = 0.0;
    double tolerance = 0.0;
    double decay = 0.0;  // size dependent tolerance (tolerance-* except legacy)
    double offset = 0.0;

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
double radius_complement(uint64_t sum_kq, uint64_t sum_kqsq, uint64_t n,
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
    uint64_t old_n = 0, nom_n = 0, new_n = 0;
    uint64_t old_sum_kq = 0, old_sum_kqsq = 0, new_sum_kq = 0, new_sum_kqsq = 0;
};

// Same as check_merge of the builtin criteria of bblean.merges
template <typename NewRc, typename OldRc>
bool check_merge(const MergeCriterion& c, const MergeSums& s, NewRc&& new_rc,
                 OldRc&& old_rc) {
    const double thr = c.threshold;
    switch (c.kind) {
        case MergeKind::kDiameter:
            return isim(s.new_sum_kq, s.new_sum_kqsq,
                        static_cast<int64_t>(s.new_n)) >= thr;
        case MergeKind::kRadius:
            return new_rc() >= thr;
        case MergeKind::kToleranceDiameter:
        case MergeKind::kFlexibleToleranceDiameter: {
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
            if (c.kind == MergeKind::kFlexibleToleranceDiameter &&
                thr < old_dc) {
                ref = thr;  // min(old_dc, thr)
            }
            return new_dc >= ref - c.tolerance_for(s.old_n);
        }
        case MergeKind::kToleranceRadius: {
            const double new_rc_value = new_rc();
            if (new_rc_value < thr) {
                return false;
            }
            if (s.old_n == 1) {
                return true;
            }
            return new_rc_value >= old_rc() - c.tolerance_for(s.old_n);
        }
        case MergeKind::kToleranceLegacy: {
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
        case MergeKind::kNever:
        default:
            return false;
    }
}

// -------------------------------------------------------------------- tree

// Linear sum of a subcluster, stored with the smallest unsigned type that holds
// its n_samples. Not stored (std::monostate) for subclusters of one, possibly
// weighted, fingerprint: it is n_samples * centroid.
using LinearSum =
    std::variant<std::monostate, std::vector<uint8_t>, std::vector<uint16_t>,
                 std::vector<uint32_t>, std::vector<uint64_t>>;

// Index in LinearSum of the smallest type that holds n
size_t linear_sum_type_index(uint64_t n) {
    if (n <= std::numeric_limits<uint8_t>::max()) {
        return 1;
    }
    if (n <= std::numeric_limits<uint16_t>::max()) {
        return 2;
    }
    if (n <= std::numeric_limits<uint32_t>::max()) {
        return 3;
    }
    return 4;
}

// count zeros, with the smallest type that holds n
LinearSum zero_linear_sum(uint64_t n, size_t count) {
    switch (linear_sum_type_index(n)) {
        case 1:
            return std::vector<uint8_t>(count);
        case 2:
            return std::vector<uint16_t>(count);
        case 3:
            return std::vector<uint32_t>(count);
        default:
            return std::vector<uint64_t>(count);
    }
}

// count values as a numpy array of the smallest uint dtype that holds n
py::array smallest_uint_array(const uint64_t* values, size_t count,
                              uint64_t n) {
    auto make = [&](auto tag) -> py::array {
        using V = decltype(tag);
        py::array_t<V> a(static_cast<py::ssize_t>(count));
        auto* p = a.mutable_data();
        for (size_t j = 0; j < count; ++j) {
            p[j] = static_cast<V>(values[j]);
        }
        return std::move(a);
    };
    switch (linear_sum_type_index(n)) {
        case 1:
            return make(uint8_t{});
        case 2:
            return make(uint16_t{});
        case 3:
            return make(uint32_t{});
        default:
            return make(uint64_t{});
    }
}

struct Subcluster {
    uint64_t n_samples = 0;
    uint64_t sum_kq = 0;    // sum of the linear sum
    uint64_t sum_kqsq = 0;  // sum of squares of the linear sum
    int32_t child = -1;
    LinearSum linear_sum;
    int64_t mol_indices_head = -1,
            mol_indices_tail =
                -1;  // molecule index chain (leaf subclusters only)
};

struct Node {
    int branching_factor = 0;
    bool is_leaf = false;
    int32_t prev_leaf = -1,
            next_leaf = -1;  // leaf chain, prev_leaf == -1 means "first leaf"
    std::vector<int32_t> subclusters;
    // Centroids of the subclusters (rows of w64 words) and their popcounts.
    // Space for branching_factor + 1 rows is reserved, but only the rows in use
    // are written, so the memory of large nodes is not touched up front.
    std::vector<uint64_t> centroids;
    std::vector<uint32_t> centroid_popcounts;
};

// A subcluster about to be inserted: either one (weighted) fingerprint, whose
// linear sum is n * fp, or a dense linear sum (uint64, internal bit order).
struct Nominee {
    // Centroid (the fingerprint itself if single)
    const uint64_t* centroid = nullptr;
    uint32_t centroid_popcount = 0;
    uint64_t n_samples = 0, sum_kq = 0, sum_kqsq = 0;
    bool dense = false;
    const uint64_t* dense_ls = nullptr;
    int64_t mol_indices_head = -1, mol_indices_tail = -1;
};

// Node and subcluster ids are int32_t, which keeps nodes small
void check_id_space(size_t count) {
    if (count >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("too many BitBirch subclusters");
    }
}

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
        scratch_sum_a_.assign(n_bits_, 0);
        scratch_sum_b_.assign(n_bits_, 0);
        scratch_centroid_.assign(w64_, 0);
        scratch_fp_.assign(w64_, 0);
    }

    int n_features() const { return n_features_; }
    int n_bytes() const { return n_bytes_; }
    // Bytes per row of 64 bit words (n_bytes_ zero padded to whole words), as
    // the kernels read them
    size_t row_bytes() const { return 8 * static_cast<size_t>(w64_); }
    bool only_has_leaves() const { return root_ < 0 && first_leaf_ >= 0; }

    // ------------------------------------------------------------- fitting

    // Fit packed fingerprints (rows of n_features / 8 bytes). Molecule indices
    // are ids[i] if given, else start + i; weights (optional) per row.
    void fit(const uint8_t* X, int64_t rows, const int64_t* ids, int64_t start,
             const uint64_t* weights, const MergeCriterion& merge,
             int branching_factor) {
        start_fit(merge, branching_factor);
        for (int64_t i = 0; i < rows; ++i) {
            uint64_t* fp = scratch_fp_.data();
            fp[w64_ - 1] = 0;
            std::memcpy(fp, X + i * n_bytes_, n_bytes_);
            const uint64_t w = weights ? weights[i] : 1;
            Nominee nominee;
            nominee.centroid = fp;
            nominee.centroid_popcount = popcount(bytes(fp), row_bytes());
            nominee.n_samples = w;
            nominee.sum_kq = w * nominee.centroid_popcount;
            nominee.sum_kqsq = w * w * nominee.centroid_popcount;
            nominee.mol_indices_head = nominee.mol_indices_tail =
                new_mol_index(ids ? ids[i] : start + i);
            insert_root(nominee, branching_factor);
        }
    }

    // Fit buffers: rows of n_features + 1 unsigned values (linear sum in
    // feature order, then n_samples). Molecule indices: idx[offs[i]:offs[i+1]]
    // (offs may be null: no indices).
    template <typename T>
    void fit_buffers(const T* X, int64_t rows, const int64_t* idx,
                     const int64_t* offs, const MergeCriterion& merge,
                     int branching_factor) {
        start_fit(merge, branching_factor);
        std::vector<uint64_t> dense_ls(n_bits_, 0);
        std::vector<uint64_t> cent(w64_, 0);
        for (int64_t i = 0; i < rows; ++i) {
            const T* buf = X + i * (n_features_ + 1);
            for (int p = 0; p < n_features_; ++p) {
                dense_ls[p] = static_cast<uint64_t>(buf[p ^ 7]);
            }
            Nominee nominee;
            nominee.n_samples = static_cast<uint64_t>(buf[n_features_]);
            centroid_and_sums(dense_ls.data(), nominee.n_samples, cent.data(),
                              nominee.centroid_popcount, nominee.sum_kq,
                              nominee.sum_kqsq);
            nominee.centroid = cent.data();
            nominee.dense = true;
            nominee.dense_ls = dense_ls.data();
            if (offs != nullptr) {
                for (int64_t k = offs[i]; k < offs[i + 1]; ++k) {
                    const int64_t m = new_mol_index(idx[k]);
                    if (nominee.mol_indices_head < 0) {
                        nominee.mol_indices_head = m;
                    } else {
                        mol_indices_next_[nominee.mol_indices_tail] = m;
                    }
                    nominee.mol_indices_tail = m;
                }
            }
            insert_root(nominee, branching_factor);
        }
    }

    void delete_internal_nodes() {
        if (root_ < 0 || nodes_[root_].is_leaf) {
            return;
        }
        std::vector<char> is_leaf(nodes_.size(), 0);
        for (int32_t l = first_leaf_; l >= 0; l = nodes_[l].next_leaf) {
            is_leaf[l] = 1;
        }
        for (size_t node_id = 0; node_id < nodes_.size(); ++node_id) {
            if (is_leaf[node_id]) {
                continue;
            }
            for (const int32_t e : nodes_[node_id].subclusters) {
                free_subcluster(e);
            }
            Node empty;
            std::swap(nodes_[node_id], empty);
        }
        root_ = -1;
    }

    // -------------------------------------------------------------- export

    // Molecule indices of the leaf subclusters
    py::list get_cluster_mol_ids(bool sort) const {
        const auto order = leaf_subclusters(sort);
        py::list out(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            py::list mol_ids;
            for (int64_t m = subclusters_[order[i]].mol_indices_head; m >= 0;
                 m = mol_indices_next_[m]) {
                mol_ids.append(mol_indices_[m]);
            }
            out[i] = mol_ids;
        }
        return out;
    }

    // Cluster label (1-based, in the order of the leaf subclusters) of each of
    // the n_mols molecules, 0 for discarded molecules
    py::array_t<uint64_t> get_assignments(int64_t n_mols, bool sort) const {
        const auto order = leaf_subclusters(sort);
        py::array_t<uint64_t> out(static_cast<py::ssize_t>(n_mols));
        auto* labels = out.mutable_data();
        std::fill(labels, labels + n_mols, 0);
        for (size_t i = 0; i < order.size(); ++i) {
            for (int64_t m = subclusters_[order[i]].mol_indices_head; m >= 0;
                 m = mol_indices_next_[m]) {
                const int64_t id = mol_indices_[m];
                if (id < 0 || id >= n_mols) {
                    throw py::index_error(
                        "molecule index out of bounds for assignments");
                }
                labels[id] = static_cast<uint64_t>(i + 1);
            }
        }
        return out;
    }

    // Packed centroids of the leaf subclusters
    py::array_t<uint8_t> get_centroids(bool sort) const {
        const auto order = leaf_subclusters(sort);
        const auto where = leaf_centroids(order);
        py::array_t<uint8_t> out({static_cast<py::ssize_t>(order.size()),
                                  static_cast<py::ssize_t>(n_bytes_)});
        auto* p = out.mutable_data();
        for (size_t i = 0; i < order.size(); ++i) {
            std::memcpy(p + i * n_bytes_, where[i], n_bytes_);
        }
        return out;
    }

    // Linear sum + n_samples buffers of the leaf subclusters, with the smallest
    // uint dtype that holds n_samples
    py::list get_leaf_buffers(bool sort) const {
        const auto order = leaf_subclusters(sort);
        const auto where = leaf_centroids(order);
        std::vector<uint64_t> internal_order(n_bits_);
        std::vector<uint64_t> buffer(n_features_ + 1);
        py::list out(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            const Subcluster& subcluster = subclusters_[order[i]];
            expand_linear_sum(subcluster, where[i], internal_order.data());
            for (int j = 0; j < n_features_; ++j) {
                buffer[j] = internal_order[j ^ 7];
            }
            buffer[n_features_] = subcluster.n_samples;
            out[i] = smallest_uint_array(buffer.data(), buffer.size(),
                                         subcluster.n_samples);
        }
        return out;
    }

    // ----------------------------------------------------------- pickling

    std::string serialize() const;
    static Tree deserialize(const std::string& s);

   private:
    // Leaf subclusters in leaf order, or sorted by n_samples (descending,
    // stable)
    std::vector<int32_t> leaf_subclusters(bool sort) const {
        std::vector<int32_t> out;
        for (int32_t l = first_leaf_; l >= 0; l = nodes_[l].next_leaf) {
            out.insert(out.end(), nodes_[l].subclusters.begin(),
                       nodes_[l].subclusters.end());
        }
        if (sort) {
            std::stable_sort(out.begin(), out.end(),
                             [this](int32_t a, int32_t b) {
                                 return subclusters_[a].n_samples >
                                        subclusters_[b].n_samples;
                             });
        }
        return out;
    }

    std::vector<const uint64_t*> leaf_centroids(
        const std::vector<int32_t>& order) const {
        std::vector<const uint64_t*> where(subclusters_.size(), nullptr);
        for (int32_t l = first_leaf_; l >= 0; l = nodes_[l].next_leaf) {
            for (size_t i = 0; i < nodes_[l].subclusters.size(); ++i) {
                where[nodes_[l].subclusters[i]] =
                    nodes_[l].centroids.data() + i * w64_;
            }
        }
        std::vector<const uint64_t*> out(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            out[i] = where[order[i]];
        }
        return out;
    }

    void start_fit(const MergeCriterion& merge, int branching_factor) {
        if (only_has_leaves()) {
            throw std::runtime_error(
                "Internal nodes were released, call reset() before fit()");
        }
        if (branching_factor < 1) {
            throw std::invalid_argument("branching_factor must be >= 1");
        }
        merge_ = merge;
        if (root_ < 0) {
            root_ = new_node(branching_factor, true);
            first_leaf_ = root_;
        }
    }

    int64_t new_mol_index(int64_t id) {
        mol_indices_.push_back(id);
        mol_indices_next_.push_back(-1);
        return static_cast<int64_t>(mol_indices_.size()) - 1;
    }

    int32_t new_node(int branching_factor, bool leaf) {
        Node node;
        node.branching_factor = branching_factor;
        node.is_leaf = leaf;
        node.subclusters.reserve(branching_factor + 1);
        node.centroids.reserve(static_cast<size_t>(branching_factor + 1) *
                               w64_);
        node.centroid_popcounts.reserve(branching_factor + 1);
        check_id_space(nodes_.size());
        nodes_.push_back(std::move(node));
        return static_cast<int32_t>(nodes_.size()) - 1;
    }

    int32_t new_subcluster() {
        if (!free_subclusters_.empty()) {
            const int32_t e = free_subclusters_.back();
            free_subclusters_.pop_back();
            subclusters_[e] = Subcluster();
            return e;
        }
        check_id_space(subclusters_.size());
        subclusters_.emplace_back();
        return static_cast<int32_t>(subclusters_.size()) - 1;
    }

    void free_subcluster(int32_t e) {
        subclusters_[e] = Subcluster();
        free_subclusters_.push_back(e);
    }

    uint64_t* centroid_row(int32_t node_id, int idx) {
        return nodes_[node_id].centroids.data() +
               static_cast<size_t>(idx) * w64_;
    }

    void set_centroid_row(int32_t node_id, int idx, const uint64_t* c,
                          uint32_t pop) {
        std::memcpy(centroid_row(node_id, idx), c, sizeof(uint64_t) * w64_);
        nodes_[node_id].centroid_popcounts[idx] = pop;
    }

    void append_subcluster(int32_t node_id, int32_t e, const uint64_t* c,
                           uint32_t pop) {
        Node& node = nodes_[node_id];
        const int idx = static_cast<int>(node.subclusters.size());
        if (idx > node.branching_factor) {
            throw std::runtime_error("node overflow");
        }
        node.subclusters.push_back(e);
        const size_t rows = static_cast<size_t>(idx) + 1;
        if (node.centroids.size() < rows * w64_) {
            node.centroids.resize(rows * w64_);
            node.centroid_popcounts.resize(rows);
        }
        set_centroid_row(node_id, idx, c, pop);
    }

    // f(get_ls) with get_ls(p) = L_p, specialized for the storage of subcluster
    template <typename F>
    void with_linear_sum(const Subcluster& subcluster, const uint64_t* cent,
                         F&& f) const {
        std::visit(
            [&](const auto& ls) {
                using T = std::decay_t<decltype(ls)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    const uint64_t n = subcluster.n_samples;
                    f([cent, n](int p) -> uint64_t {
                        return ((cent[p >> 6] >> (p & 63)) & 1) ? n : 0;
                    });
                } else {
                    const auto* values = ls.data();
                    f([values](int p) -> uint64_t { return values[p]; });
                }
            },
            subcluster.linear_sum);
    }

    // Dense linear sum into out (uint64, internal order)
    void expand_linear_sum(const Subcluster& subcluster, const uint64_t* cent,
                           uint64_t* out) const {
        with_linear_sum(subcluster, cent, [&](auto get_ls) {
            for (int p = 0; p < n_bits_; ++p) {
                out[p] = get_ls(p);
            }
        });
    }

    // Centroid (with its popcount), sum_kq and sum_kqsq of a dense linear sum
    // of n_samples fingerprints
    void centroid_and_sums(const uint64_t* ls, uint64_t n_samples,
                           uint64_t* cent, uint32_t& pop, uint64_t& sum_kq,
                           uint64_t& sum_kqsq) const {
        std::fill(cent, cent + w64_, 0);
        sum_kq = sum_kqsq = 0;
        for (int p = 0; p < n_bits_; ++p) {
            sum_kq += ls[p];
            sum_kqsq += ls[p] * ls[p];
            if (centroid_value(ls[p], static_cast<int64_t>(n_samples)) != 0) {
                cent[p >> 6] |= uint64_t{1} << (p & 63);
            }
        }
        pop = popcount(bytes(cent), row_bytes());
    }

    // Store a dense linear sum with the smallest type for n_samples n
    void store_linear_sum(Subcluster& subcluster, const uint64_t* dense,
                          uint64_t n) {
        subcluster.linear_sum = zero_linear_sum(n, n_bits_);
        std::visit(
            [&](auto& ls) {
                using T = std::decay_t<decltype(ls)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    using V = typename T::value_type;
                    for (int p = 0; p < n_bits_; ++p) {
                        ls[p] = static_cast<V>(dense[p]);
                    }
                }
            },
            subcluster.linear_sum);
    }

    // Make sure the subcluster stores its linear sum, with a type that holds
    // new_n
    void widen_linear_sum(Subcluster& subcluster, const uint64_t* cent,
                          uint64_t new_n) {
        if (subcluster.linear_sum.index() >= linear_sum_type_index(new_n)) {
            return;
        }
        uint64_t* tmp = scratch_sum_b_.data();
        expand_linear_sum(subcluster, cent, tmp);
        store_linear_sum(subcluster, tmp, new_n);
    }

    // Index of the closest subcluster (first maximum of the Tanimoto
    // similarity)
    int closest_idx(int32_t node_id, const Nominee& nominee) {
        const Node& node = nodes_[node_id];
        const size_t m = node.subclusters.size();
        sims_a_.resize(m);
        jt_sims(bytes(node.centroids.data()), node.centroid_popcounts.data(), m,
                row_bytes(), bytes(nominee.centroid), sims_a_.data());
        return static_cast<int>(
            std::max_element(sims_a_.begin(), sims_a_.end()) - sims_a_.begin());
    }

    // New centroid of subcluster (centroid old_cent) after adding nominee; also
    // returns cent_sum = sum_{j in new centroid} ls'_j (ls' = merged linear
    // sum), new_sum_kq and new_sum_kqsq.
    void merged_centroid_and_sums(const Subcluster& subcluster,
                                  const uint64_t* old_cent,
                                  const Nominee& nominee, uint64_t* new_cent,
                                  uint32_t& new_pop, uint64_t& cent_sum,
                                  uint64_t& new_sum_kq,
                                  uint64_t& new_sum_kqsq) {
        const uint64_t new_n = subcluster.n_samples + nominee.n_samples;
        uint64_t cross = 0;
        cent_sum = 0;
        if (!nominee.dense) {
            // ls'_j = ls_j + nominee_n * fp_j; only bits in old_cent | fp can
            // be set in the new centroid
            with_linear_sum(subcluster, old_cent, [&](auto get_ls) {
                const uint64_t nominee_n = nominee.n_samples;
                for (int k = 0; k < w64_; ++k) {
                    const uint64_t f = nominee.centroid[k];
                    uint64_t out = 0;
                    for (uint64_t w = old_cent[k] | f; w; w &= w - 1) {
                        const int b = count_trailing_zeros(w);
                        const uint64_t old_value = get_ls(k * 64 + b);
                        const uint64_t in_fp = (f >> b) & 1;
                        cross += in_fp * old_value;
                        const uint64_t new_value =
                            old_value + in_fp * nominee_n;
                        if (centroid_value(new_value, new_n) != 0) {
                            out |= uint64_t{1} << b;
                            cent_sum += new_value;
                        }
                    }
                    new_cent[k] = out;
                }
            });
            cross *= nominee.n_samples;
        } else {
            uint64_t* ls = scratch_sum_a_.data();
            expand_linear_sum(subcluster, old_cent, ls);
            std::fill(new_cent, new_cent + w64_, 0);
            for (int p = 0; p < n_bits_; ++p) {
                cross += ls[p] * nominee.dense_ls[p];
                const uint64_t new_value = ls[p] + nominee.dense_ls[p];
                if (centroid_value(new_value, new_n) != 0) {
                    new_cent[p >> 6] |= uint64_t{1} << (p & 63);
                    cent_sum += new_value;
                }
            }
        }
        new_pop = popcount(bytes(new_cent), row_bytes());
        new_sum_kq = subcluster.sum_kq + nominee.sum_kq;
        new_sum_kqsq = subcluster.sum_kqsq + nominee.sum_kqsq + 2 * cross;
    }

    bool accepts_merge(const Subcluster& subcluster, const uint64_t* old_cent,
                       uint32_t old_pop, const Nominee& nominee,
                       uint64_t new_sum_kq, uint64_t new_sum_kqsq,
                       uint32_t new_pop, uint64_t cent_sum) const {
        const MergeSums sums{subcluster.n_samples,
                             nominee.n_samples,
                             subcluster.n_samples + nominee.n_samples,
                             subcluster.sum_kq,
                             subcluster.sum_kqsq,
                             new_sum_kq,
                             new_sum_kqsq};
        return check_merge(
            merge_, sums,
            [&] {
                return radius_complement(
                    new_sum_kq, new_sum_kqsq,
                    subcluster.n_samples + nominee.n_samples, new_pop,
                    cent_sum);
            },
            [&] {
                uint64_t old_cent_sum = 0;
                with_linear_sum(subcluster, old_cent, [&](auto get_ls) {
                    for_each_set_bit(bytes(old_cent), row_bytes(),
                                     [&](int p) { old_cent_sum += get_ls(p); });
                });
                return radius_complement(subcluster.sum_kq, subcluster.sum_kqsq,
                                         subcluster.n_samples, old_pop,
                                         old_cent_sum);
            });
    }

    // Add the nominee to the subcluster at (node_id, idx), with precomputed
    // stats
    void merge_subcluster(int32_t node_id, int idx, const Nominee& nominee,
                          const uint64_t* new_cent, uint32_t new_pop,
                          uint64_t new_sum_kq, uint64_t new_sum_kqsq,
                          bool with_mol_indices) {
        Subcluster& subcluster = subclusters_[nodes_[node_id].subclusters[idx]];
        const uint64_t new_n = subcluster.n_samples + nominee.n_samples;
        widen_linear_sum(subcluster, centroid_row(node_id, idx), new_n);
        std::visit(
            [&](auto& ls) {
                using T = std::decay_t<decltype(ls)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    using V = typename T::value_type;
                    if (!nominee.dense) {
                        const V add = static_cast<V>(nominee.n_samples);
                        for_each_set_bit(
                            bytes(nominee.centroid), row_bytes(), [&](int p) {
                                ls[p] = static_cast<V>(ls[p] + add);
                            });
                    } else {
                        for (int p = 0; p < n_bits_; ++p) {
                            ls[p] = static_cast<V>(ls[p] + nominee.dense_ls[p]);
                        }
                    }
                }
            },
            subcluster.linear_sum);
        subcluster.n_samples = new_n;
        subcluster.sum_kq = new_sum_kq;
        subcluster.sum_kqsq = new_sum_kqsq;
        if (with_mol_indices && nominee.mol_indices_head >= 0) {
            if (subcluster.mol_indices_head < 0) {
                subcluster.mol_indices_head = nominee.mol_indices_head;
            } else {
                mol_indices_next_[subcluster.mol_indices_tail] =
                    nominee.mol_indices_head;
            }
            subcluster.mol_indices_tail = nominee.mol_indices_tail;
        }
        set_centroid_row(node_id, idx, new_cent, new_pop);
    }

    void append_nominee(int32_t node_id, const Nominee& nominee) {
        const int32_t e = new_subcluster();
        Subcluster& subcluster = subclusters_[e];
        subcluster.n_samples = nominee.n_samples;
        subcluster.sum_kq = nominee.sum_kq;
        subcluster.sum_kqsq = nominee.sum_kqsq;
        subcluster.mol_indices_head = nominee.mol_indices_head;
        subcluster.mol_indices_tail = nominee.mol_indices_tail;
        if (nominee.dense) {
            store_linear_sum(subcluster, nominee.dense_ls,
                             subcluster.n_samples);
        }
        append_subcluster(node_id, e, nominee.centroid,
                          nominee.centroid_popcount);
    }

    struct SplitResult {
        int32_t subcluster1 = -1,
                subcluster2 = -1;  // tracking subclusters of the two nodes
        std::vector<uint64_t> centroid1, centroid2;
        uint32_t pop1 = 0, pop2 = 0;
    };

    SplitResult split(int32_t node_id) {
        const int m = static_cast<int>(nodes_[node_id].subclusters.size());
        const bool leaf = nodes_[node_id].is_leaf;
        const int branching_factor = nodes_[node_id].branching_factor;
        const std::vector<int32_t> subclusters = nodes_[node_id].subclusters;
        const std::vector<uint64_t> rows(
            nodes_[node_id].centroids.begin(),
            nodes_[node_id].centroids.begin() + static_cast<size_t>(m) * w64_);
        const std::vector<uint32_t> pops(
            nodes_[node_id].centroid_popcounts.begin(),
            nodes_[node_id].centroid_popcounts.begin() + m);
        auto row_of = [&](int i) {
            return rows.data() + static_cast<size_t>(i) * w64_;
        };

        sims_a_.resize(m);
        sims_b_.resize(m);
        const size_t fp1 =
            most_dissimilar(bytes(rows.data()), pops.data(), m, row_bytes(),
                            n_bits_, sims_a_.data(), sims_b_.data())
                .fp1;

        // node1 is new, node2 is the old node (reused)
        const int32_t node1_id = new_node(branching_factor, leaf);
        if (leaf) {
            Node& node2 = nodes_[node_id];
            Node& node1 = nodes_[node1_id];
            node1.prev_leaf = node2.prev_leaf;
            if (node2.prev_leaf >= 0) {
                nodes_[node2.prev_leaf].next_leaf = node1_id;
            } else {
                first_leaf_ = node1_id;
            }
            node1.next_leaf = node_id;
            node2.prev_leaf = node1_id;
        }
        nodes_[node_id].subclusters.clear();
        std::vector<uint64_t> acc1(n_bits_, 0), acc2(n_bits_, 0);
        uint64_t n_samples1 = 0, n_samples2 = 0;
        uint64_t* tmp = scratch_sum_a_.data();
        for (int i = 0; i < m; ++i) {
            const bool closer1 =
                static_cast<size_t>(i) == fp1 || sims_a_[i] > sims_b_[i];
            const Subcluster& subcluster = subclusters_[subclusters[i]];
            std::vector<uint64_t>& acc = closer1 ? acc1 : acc2;
            expand_linear_sum(subcluster, row_of(i), tmp);
            for (int p = 0; p < n_bits_; ++p) {
                acc[p] += tmp[p];
            }
            if (closer1) {
                n_samples1 += subcluster.n_samples;
                append_subcluster(node1_id, subclusters[i], row_of(i), pops[i]);
            } else {
                n_samples2 += subcluster.n_samples;
                append_subcluster(node_id, subclusters[i], row_of(i), pops[i]);
            }
        }
        SplitResult out;
        out.subcluster1 = new_tracking_subcluster(node1_id, n_samples1, acc1,
                                                  out.centroid1, out.pop1);
        out.subcluster2 = new_tracking_subcluster(node_id, n_samples2, acc2,
                                                  out.centroid2, out.pop2);
        return out;
    }

    int32_t new_tracking_subcluster(int32_t child, uint64_t n,
                                    const std::vector<uint64_t>& acc,
                                    std::vector<uint64_t>& c, uint32_t& pop) {
        const int32_t e = new_subcluster();
        Subcluster& subcluster = subclusters_[e];
        subcluster.child = child;
        subcluster.n_samples = n;
        c.resize(w64_);
        centroid_and_sums(acc.data(), n, c.data(), pop, subcluster.sum_kq,
                          subcluster.sum_kqsq);
        store_linear_sum(subcluster, acc.data(), n);
        return e;
    }

    // Insert the nominee below node node_id; returns whether it must be split
    bool insert_bf_subcluster(int32_t node_id, const Nominee& nominee) {
        if (nodes_[node_id].subclusters.empty()) {
            append_nominee(node_id, nominee);
            return false;
        }
        const int idx = closest_idx(node_id, nominee);
        const int32_t e = nodes_[node_id].subclusters[idx];
        uint64_t* new_cent = scratch_centroid_.data();
        uint32_t new_pop = 0;
        uint64_t cent_sum = 0, new_sum_kq = 0, new_sum_kqsq = 0;
        if (subclusters_[e].child < 0) {
            const uint64_t* old_cent = centroid_row(node_id, idx);
            merged_centroid_and_sums(subclusters_[e], old_cent, nominee,
                                     new_cent, new_pop, cent_sum, new_sum_kq,
                                     new_sum_kqsq);
            if (accepts_merge(subclusters_[e], old_cent,
                              nodes_[node_id].centroid_popcounts[idx], nominee,
                              new_sum_kq, new_sum_kqsq, new_pop, cent_sum)) {
                merge_subcluster(node_id, idx, nominee, new_cent, new_pop,
                                 new_sum_kq, new_sum_kqsq, true);
                return false;
            }
            append_nominee(node_id, nominee);
            return static_cast<int>(nodes_[node_id].subclusters.size()) >
                   nodes_[node_id].branching_factor;
        }
        const int32_t child = subclusters_[e].child;
        if (insert_bf_subcluster(child, nominee)) {
            SplitResult split_result = split(child);
            nodes_[node_id].subclusters[idx] = split_result.subcluster1;
            set_centroid_row(node_id, idx, split_result.centroid1.data(),
                             split_result.pop1);
            free_subcluster(e);
            append_subcluster(node_id, split_result.subcluster2,
                              split_result.centroid2.data(), split_result.pop2);
            return static_cast<int>(nodes_[node_id].subclusters.size()) >
                   nodes_[node_id].branching_factor;
        }
        // Update the tracking subcluster
        merged_centroid_and_sums(subclusters_[e], centroid_row(node_id, idx),
                                 nominee, new_cent, new_pop, cent_sum,
                                 new_sum_kq, new_sum_kqsq);
        merge_subcluster(node_id, idx, nominee, new_cent, new_pop, new_sum_kq,
                         new_sum_kqsq, false);
        return false;
    }

    void insert_root(const Nominee& nominee, int branching_factor) {
        if (insert_bf_subcluster(root_, nominee)) {
            SplitResult split_result = split(root_);
            const int32_t r = new_node(branching_factor, false);
            append_subcluster(r, split_result.subcluster1,
                              split_result.centroid1.data(), split_result.pop1);
            append_subcluster(r, split_result.subcluster2,
                              split_result.centroid2.data(), split_result.pop2);
            root_ = r;
        }
    }

    int n_features_ = 0, n_bytes_ = 0, w64_ = 0, n_bits_ = 0;
    MergeCriterion merge_{};
    std::vector<Node> nodes_;
    std::vector<Subcluster> subclusters_;
    std::vector<int32_t> free_subclusters_;
    std::vector<int64_t> mol_indices_, mol_indices_next_;
    int32_t root_ = -1, first_leaf_ = -1;
    // scratch
    std::vector<uint64_t> scratch_sum_a_, scratch_sum_b_, scratch_centroid_,
        scratch_fp_;
    std::vector<double> sims_a_, sims_b_;
};

// ---------------------------------------------------------------- pickling

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
    std::vector<T> vec() {
        std::vector<T> v;
        vec(v);
        return v;
    }
    LinearSum linear_sum(uint8_t index) {
        switch (index) {
            case 0:
                return std::monostate{};
            case 1:
                return vec<uint8_t>();
            case 2:
                return vec<uint16_t>();
            case 3:
                return vec<uint32_t>();
            case 4:
                return vec<uint64_t>();
            default:
                throw std::runtime_error("corrupt BitBirch tree state");
        }
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

std::string Tree::serialize() const {
    Writer w;
    w.put(kMagic);
    w.put<int32_t>(n_features_);
    w.put<int32_t>(root_);
    w.put<int32_t>(first_leaf_);
    w.vec(mol_indices_);
    w.vec(mol_indices_next_);
    w.vec(free_subclusters_);
    w.put<uint64_t>(nodes_.size());
    for (const Node& node : nodes_) {
        w.put<int32_t>(node.branching_factor);
        w.put<uint8_t>(node.is_leaf);
        w.put<int32_t>(node.prev_leaf);
        w.put<int32_t>(node.next_leaf);
        w.vec(node.subclusters);
        w.vec(node.centroids);
        w.vec(node.centroid_popcounts);
    }
    w.put<uint64_t>(subclusters_.size());
    for (const Subcluster& subcluster : subclusters_) {
        w.put(subcluster.n_samples);
        w.put(subcluster.sum_kq);
        w.put(subcluster.sum_kqsq);
        w.put(subcluster.child);
        w.put<uint8_t>(static_cast<uint8_t>(subcluster.linear_sum.index()));
        w.put(subcluster.mol_indices_head);
        w.put(subcluster.mol_indices_tail);
        std::visit(
            [&](const auto& ls) {
                if constexpr (!std::is_same_v<std::decay_t<decltype(ls)>,
                                              std::monostate>) {
                    w.vec(ls);
                }
            },
            subcluster.linear_sum);
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
    r.vec(t.mol_indices_);
    r.vec(t.mol_indices_next_);
    r.vec(t.free_subclusters_);
    t.nodes_.resize(r.get<uint64_t>());
    for (Node& node : t.nodes_) {
        node.branching_factor = r.get<int32_t>();
        node.is_leaf = r.get<uint8_t>() != 0;
        node.prev_leaf = r.get<int32_t>();
        node.next_leaf = r.get<int32_t>();
        r.vec(node.subclusters);
        r.vec(node.centroids);
        r.vec(node.centroid_popcounts);
    }
    t.subclusters_.resize(r.get<uint64_t>());
    for (Subcluster& subcluster : t.subclusters_) {
        subcluster.n_samples = r.get<uint64_t>();
        subcluster.sum_kq = r.get<uint64_t>();
        subcluster.sum_kqsq = r.get<uint64_t>();
        subcluster.child = r.get<int32_t>();
        const uint8_t linear_sum_index = r.get<uint8_t>();
        subcluster.mol_indices_head = r.get<int64_t>();
        subcluster.mol_indices_tail = r.get<int64_t>();
        subcluster.linear_sum = r.linear_sum(linear_sum_index);
    }
    return t;
}

// ----------------------------------------------------------------- bindings

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
    static const std::unordered_map<std::string, MergeKind> kinds{
        {"diameter", MergeKind::kDiameter},
        {"radius", MergeKind::kRadius},
        {"tolerance-diameter", MergeKind::kToleranceDiameter},
        {"flexible-tolerance-diameter", MergeKind::kFlexibleToleranceDiameter},
        {"tolerance-radius", MergeKind::kToleranceRadius},
        {"tolerance-legacy", MergeKind::kToleranceLegacy},
        {"never", MergeKind::kNever},
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

void fit(Tree& tree, const py::array_t<uint8_t, py::array::c_style>& X,
         const py::object& ids, int64_t start, const py::object& weights,
         const py::tuple& merge, int branching_factor) {
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
    tree.fit(X.data(), rows, id_ptr, start, w_ptr, m, branching_factor);
}

void fit_buffers(Tree& tree, const py::array& X, const py::object& idx,
                 const py::object& offs, const py::tuple& merge,
                 int branching_factor) {
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
    switch (size) {
        case 1:
            return tree.fit_buffers(static_cast<const uint8_t*>(data), rows,
                                    id_ptr, off_ptr, m, branching_factor);
        case 2:
            return tree.fit_buffers(static_cast<const uint16_t*>(data), rows,
                                    id_ptr, off_ptr, m, branching_factor);
        case 4:
            return tree.fit_buffers(static_cast<const uint32_t*>(data), rows,
                                    id_ptr, off_ptr, m, branching_factor);
        default:
            return tree.fit_buffers(static_cast<const uint64_t*>(data), rows,
                                    id_ptr, off_ptr, m, branching_factor);
    }
}

}  // namespace
}  // namespace bblean

PYBIND11_MODULE(_cpp_bitbirch, m) {
    using bblean::Tree;
    m.doc() = "BitBirch tree (C++ extension)";
    py::class_<Tree>(m, "Tree")
        .def(py::init<int>(), py::arg("n_features"))
        .def_property_readonly("n_features", &Tree::n_features)
        .def_property_readonly("only_has_leaves", &Tree::only_has_leaves)
        .def("fit", &bblean::fit, py::arg("X"), py::arg("ids"),
             py::arg("start"), py::arg("weights"), py::arg("merge"),
             py::arg("branching_factor"))
        .def("fit_buffers", &bblean::fit_buffers, py::arg("X"), py::arg("idx"),
             py::arg("offsets"), py::arg("merge"), py::arg("branching_factor"))
        .def("delete_internal_nodes", &Tree::delete_internal_nodes)
        .def("get_cluster_mol_ids", &Tree::get_cluster_mol_ids,
             py::arg("sort") = true)
        .def("get_assignments", &Tree::get_assignments, py::arg("n_mols"),
             py::arg("sort") = true)
        .def("get_centroids", &Tree::get_centroids, py::arg("sort") = true)
        .def("get_leaf_buffers", &Tree::get_leaf_buffers,
             py::arg("sort") = true)
        .def(py::pickle([](const Tree& t) { return py::bytes(t.serialize()); },
                        [](const py::bytes& b) {
                            return Tree::deserialize(std::string(b));
                        }));
}
