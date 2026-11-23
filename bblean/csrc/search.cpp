// Persistent search kernels. Adapted from the November 2026 clustered-search
// experiments; uses current main's portable popcount/NEON kernels and pybind11.
// No runtime compilation, fixed fingerprint width, OpenMP or AWS dependency.
#include "kernels.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace py = pybind11;
template <typename T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

namespace {
struct Hit { double score; uint64_t id; };
bool better(const Hit& a, const Hit& b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

py::tuple topk(Array<uint8_t> fps, Array<uint64_t> ids, Array<uint32_t> pops,
               Array<uint8_t> query, size_t k, double threshold, bool exclude) {
    if (fps.ndim() != 2 || query.ndim() != 1 || fps.shape(1) != query.size() ||
        ids.ndim() != 1 || pops.ndim() != 1 || fps.shape(0) != ids.size() ||
        fps.shape(0) != pops.size() || query.size() <= 0 ||
        query.size() > std::numeric_limits<uint32_t>::max() / 8 ||
        !std::isfinite(threshold) || threshold < 0 || threshold > 1)
        throw std::invalid_argument("Invalid packed search columns/query");
    k = std::min(k, static_cast<size_t>(ids.size()));
    std::vector<Hit> heap;
    heap.reserve(k);
    if (k) {
        py::gil_scoped_release release;
        const uint32_t qp = bblean::popcount(query.data(), query.size());
        bblean::for_each_intersection_count(fps.data(), fps.shape(0), query.size(), query.data(),
            [&](size_t i, uint32_t intersection) {
                const uint64_t uni = uint64_t(qp) + pops.data()[i] - intersection;
                const double score = uni ? double(intersection) / double(uni) : 0.0;
                if ((exclude && score == 1.0) || score < threshold) return;
                Hit hit{score, ids.data()[i]};
                if (heap.size() < k) {
                    heap.push_back(hit);
                    std::push_heap(heap.begin(), heap.end(), better);
                } else if (better(hit, heap.front())) {
                    std::pop_heap(heap.begin(), heap.end(), better);
                    heap.back() = hit;
                    std::push_heap(heap.begin(), heap.end(), better);
                }
            });
        std::sort(heap.begin(), heap.end(), better);
    }
    py::array_t<uint64_t> out_ids(heap.size());
    py::array_t<double> scores(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        out_ids.mutable_data()[i] = heap[i].id;
        scores.mutable_data()[i] = heap[i].score;
    }
    return py::make_tuple(out_ids, scores);
}

py::array_t<double> route(Array<uint8_t> query, Array<uint64_t> bit_offsets,
        Array<uint64_t> code_offsets, Array<uint16_t> deltas, Array<uint8_t> codes,
        Array<float> lut, Array<float> means, Array<uint64_t> rows,
        Array<uint32_t> lists, size_t n_lists) {
    if (query.ndim() != 1 || !query.size() || means.ndim() != 1 ||
        rows.ndim() != 1 || lists.ndim() != 1 || rows.size() != means.size() ||
        lists.size() != means.size() || bit_offsets.ndim() != 1 ||
        code_offsets.ndim() != 1 || bit_offsets.size() != code_offsets.size() ||
        bit_offsets.size() < 2 || bit_offsets.size() > query.size() * 8 + 1 ||
        lut.ndim() != 1 || lut.size() == 0 || deltas.ndim() != 1 || codes.ndim() != 1 || !n_lists)
        throw std::invalid_argument("Invalid sparse routing arrays");
    const size_t units = means.size();
    py::array_t<double> result(n_lists);
    auto* best = result.mutable_data();
    {
        py::gil_scoped_release release;
        std::fill(best, best + n_lists, -std::numeric_limits<double>::infinity());
        std::vector<float> mu(units, 0), var(units, 0);
        for (size_t byte = 0; byte < static_cast<size_t>(query.size()); ++byte) {
            for (size_t shift = 0; shift < 8; ++shift) {
                if (!(query.data()[byte] & (0x80 >> shift))) continue;
                const size_t bit = byte * 8 + shift;
                if (bit + 1 >= static_cast<size_t>(bit_offsets.size()))
                    throw std::invalid_argument("Nonzero fingerprint padding bits");
                uint64_t at = bit_offsets.data()[bit], end = bit_offsets.data()[bit + 1];
                uint64_t ca = code_offsets.data()[bit], ce = code_offsets.data()[bit + 1];
                if (end > static_cast<uint64_t>(deltas.size()) || at > end ||
                    ce > static_cast<uint64_t>(codes.size()) || ca > ce)
                    throw std::invalid_argument("Invalid sparse posting offsets");
                uint64_t unit = 0;
                bool first = true;
                while (at < end) {
                    uint64_t gap = deltas.data()[at++];
                    if (gap == 65535) {
                        if (end - at < 2) throw std::invalid_argument("Truncated posting escape");
                        gap = uint64_t(deltas.data()[at]) | (uint64_t(deltas.data()[at + 1]) << 16);
                        at += 2;
                    }
                    if ((!first && !gap) || unit + gap >= units || ca >= ce)
                        throw std::invalid_argument("Invalid sparse posting ID/count");
                    first = false;
                    unit += gap;
                    const uint8_t code = codes.data()[ca++];
                    if (code >= lut.size()) throw std::invalid_argument("Invalid sparse frequency code");
                    const float f = lut.data()[code];
                    mu[unit] += f;
                    var[unit] += f * (1.0f - f);
                }
                if (ca != ce) throw std::invalid_argument("Sparse posting/code lengths disagree");
            }
        }
        const double qp = bblean::popcount(query.data(), query.size());
        for (size_t u = 0; u < units; ++u) {
            if (lists.data()[u] >= n_lists || !rows.data()[u])
                throw std::invalid_argument("Invalid summary list/population");
            const float log_term = static_cast<float>(std::sqrt(2 * std::log(double(rows.data()[u]))));
            const double x = std::min(qp, double(mu[u]) + (2 + .75 * double(log_term)) * std::sqrt(std::max(0.0, double(var[u]))));
            const double uni = qp + means.data()[u] - x;
            const double score = uni > 0 ? x / uni : 0;
            best[lists.data()[u]] = std::max(best[lists.data()[u]], score);
        }
    }
    return result;
}

py::array_t<uint32_t> split_summaries(Array<uint8_t> fps, size_t target) {
    if (fps.ndim() != 2 || fps.shape(1) <= 0 || !target ||
        fps.shape(1) > std::numeric_limits<uint32_t>::max() / 8)
        throw std::invalid_argument("Invalid summary splitting input");
    const size_t n = fps.shape(0), width = fps.shape(1), bits = width * 8;
    const size_t k = n / target + (n % target != 0);
    if (k > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Too many summary centers");
    py::array_t<uint32_t> result(n);
    auto* labels = result.mutable_data();
    if (n <= target) {
        std::fill(labels, labels + n, 0);
        return result;
    }
    {
        py::gil_scoped_release release;
        std::fill(labels, labels + n, 0);
        std::vector<uint8_t> centers(k * width);
        std::copy(fps.data(), fps.data() + width, centers.data());
        const size_t sample_n = std::min(n, k > n / 64 ? n : k * 64);
        const size_t step = std::max<size_t>(1, n / sample_n);
        std::vector<double> nearest(sample_n, -1);
        for (size_t c = 1; c < k; ++c) {
            const auto* center = centers.data() + (c - 1) * width;
            const uint32_t cp = bblean::popcount(center, width);
            size_t pick = 0;
            double worst = 2;
            for (size_t s = 0; s < sample_n; ++s) {
                const auto* row = fps.data() + s * step * width;
                uint32_t inter = 0;
                bblean::for_each_intersection_count(row, 1, width, center,
                    [&](size_t, uint32_t x) { inter = x; });
                const uint64_t uni = uint64_t(cp) + bblean::popcount(row, width) - inter;
                nearest[s] = std::max(nearest[s], uni ? double(inter) / uni : 1.0);
                if (nearest[s] < worst) { worst = nearest[s]; pick = s; }
            }
            std::copy(fps.data() + pick * step * width, fps.data() + (pick * step + 1) * width, centers.data() + c * width);
        }
        std::vector<uint64_t> tally(k * bits), populations(k);
        std::vector<double> best(n);
        for (int pass = 0; pass < 2; ++pass) {
            std::fill(best.begin(), best.end(), -1);
            std::fill(tally.begin(), tally.end(), 0);
            std::fill(populations.begin(), populations.end(), 0);
            for (size_t c = 0; c < k; ++c) {
                const auto* center = centers.data() + c * width;
                const uint32_t cp = bblean::popcount(center, width);
                bblean::for_each_intersection_count(fps.data(), n, width, center,
                    [&](size_t i, uint32_t inter) {
                        const uint64_t uni = uint64_t(cp) + bblean::popcount(fps.data() + i * width, width) - inter;
                        const double score = uni ? double(inter) / uni : 1.0;
                        if (score > best[i]) { best[i] = score; labels[i] = static_cast<uint32_t>(c); }
                    });
            }
            for (size_t i = 0; i < n; ++i) {
                const size_t c = labels[i];
                ++populations[c];
                const auto* row = fps.data() + i * width;
                for (size_t b = 0; b < bits; ++b)
                    if (row[b / 8] & (0x80 >> (b % 8))) ++tally[c * bits + b];
            }
            for (size_t c = 0; c < k; ++c) {
                if (!populations[c]) continue;
                std::fill(centers.data() + c * width, centers.data() + (c + 1) * width, 0);
                for (size_t b = 0; b < bits; ++b)
                    if (2 * tally[c * bits + b] >= populations[c])
                        centers[c * width + b / 8] |= uint8_t(0x80 >> (b % 8));
            }
        }
    }
    return result;
}
}  // namespace

PYBIND11_MODULE(_cpp_search, m) {
    m.doc() = "Packed exact top-k, sparse routing and deterministic search summaries";
    m.def("topk", &topk);
    m.def("route", &route);
    m.def("split_summaries", &split_summaries);
}
