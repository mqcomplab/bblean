// Kernels of the similarity calculations (similarity.cpp): popcounts, Tanimoto
// similarity, iSIM, majority centroids and the most dissimilar pair of a set of
// fingerprints
//
// Fingerprints are rows of 64 bit words: a packed (uint8) fingerprint viewed as
// 64 bit words, zero padded to a whole number of words. The order of the bits
// within the words does not matter for these calculations.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>  // _BitScanForward64
#endif

// The rows of 64 bit words are views of the packed bytes, which places the
// first bytes of a fingerprint in the lowest bits of the words (e.g. for the
// features that are counted in most_dissimilar)
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "Big endian architectures are not supported"
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#define BBLEAN_NEON 1
#include <arm_neon.h>
#endif

// Scalar popcount intrinsics:
#if defined(__SSE4_2__) || defined(_M_SSE4_2)
// Compiler-portable, but *not available in systems that do not have SSE*
// (which should be almost no CPUs nowadays)
// Not actually vector instructions, they just live in the SSE header
// Should be *exactly as fast* as __(builtin_)popcnt(ll) (compile to the same
// code)
//
// nmmintrin.h is the SSE4.2 intrinsics (only) header for all compilers
// NOTE: This ifdef is probably overkill, almost all cases should be covered by
// the GCC|Clang|MSVC ifdefs, but it doesn't hurt to add it
#include <nmmintrin.h>
#define POPCOUNT_32 _mm_popcnt_u32
#define POPCOUNT_64 _mm_popcnt_u64
#elif defined(_MSC_VER)
// Windows (MSVC compiler)
#include <intrin.h>
#define POPCOUNT_32 __popcnt
#define POPCOUNT_64 __popcnt64
#elif defined(__GNUC__) || defined(__clang__)
// GCC | Clang
#define POPCOUNT_32 __builtin_popcount
#define POPCOUNT_64 __builtin_popcountll
#else
// If popcnt is not hardware supported numpy rolls out its own hand-coded
// version, fail for simplicity since it is not worth it to support those archs
#error "Popcount not supported in target architecture"
#endif

namespace bblean {

inline uint32_t popcount(const uint64_t* a, size_t w64) {
    uint32_t s = 0;
    for (size_t k = 0; k < w64; ++k) {
        s += static_cast<uint32_t>(POPCOUNT_64(a[k]));
    }
    return s;
}

#if defined(_MSC_VER) && !defined(__clang__)
inline int ctz64(uint64_t x) {
    unsigned long i;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
}
#else
inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

// Calls f(p) for every set bit p of a row
template <typename F>
inline void for_each_bit(const uint64_t* a, size_t w64, F&& f) {
    for (size_t k = 0; k < w64; ++k) {
        for (uint64_t w = a[k]; w; w &= w - 1) {
            f(static_cast<int>(k * 64) + ctz64(w));
        }
    }
}

#if defined(BBLEAN_NEON)
// Query of V 16-byte vectors held in registers. Byte counts accumulate in u8
// lanes, at most 8 * V / 2 <= 128 per lane for V <= 32.
template <int V>
struct NeonQuery {
    uint8x16_t v[V];
    explicit NeonQuery(const uint64_t* q) {
        const auto* p = reinterpret_cast<const uint8_t*>(q);
        for (int k = 0; k < V; ++k) {
            v[k] = vld1q_u8(p + 16 * k);
        }
    }
    uint32_t and_popcount(const uint64_t* a) const {
        const auto* p = reinterpret_cast<const uint8_t*>(a);
        uint8x16_t acc0 = vcntq_u8(vandq_u8(vld1q_u8(p), v[0]));
        uint8x16_t acc1 = vcntq_u8(vandq_u8(vld1q_u8(p + 16), v[1]));
        for (int k = 2; k < V; k += 2) {
            acc0 =
                vaddq_u8(acc0, vcntq_u8(vandq_u8(vld1q_u8(p + 16 * k), v[k])));
            acc1 = vaddq_u8(
                acc1, vcntq_u8(vandq_u8(vld1q_u8(p + 16 * k + 16), v[k + 1])));
        }
        return vaddlvq_u8(vaddq_u8(acc0, acc1));
    }
};
#endif

// Calls f(i, popcount(row_i & query)) for each of the n rows (rows of w64
// words)
template <typename F>
inline void for_each_intersection(const uint64_t* rows, size_t n, size_t w64,
                                  const uint64_t* query, F&& f) {
#if defined(BBLEAN_NEON)
    // 2048 and 1024 bit fingerprints: the query is kept in registers
    if (w64 == 32 || w64 == 16) {
        auto scan = [&](const auto& q) {
            for (size_t i = 0; i < n; ++i) {
                f(i, q.and_popcount(rows + i * w64));
            }
        };
        if (w64 == 32) {
            scan(NeonQuery<16>(query));
        } else {
            scan(NeonQuery<8>(query));
        }
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) {
        const uint64_t* r = rows + i * w64;
        uint32_t s = 0;
        for (size_t k = 0; k < w64; ++k) {
            s += static_cast<uint32_t>(POPCOUNT_64(r[k] & query[k]));
        }
        f(i, s);
    }
}

// Tanimoto similarity from the popcounts of a & b, a and b. The similarity of
// two empty fingerprints is 0.
inline double jt_sim(uint32_t intersection, uint32_t pop_a, uint32_t pop_b) {
    const uint32_t denominator = pop_a + pop_b - intersection;
    return intersection / std::max(static_cast<double>(denominator), 1.0);
}

// Tanimoto similarities of n rows (with popcounts pops) to query
inline void jt_sims(const uint64_t* rows, const uint32_t* pops, size_t n,
                    size_t w64, const uint64_t* query, double* out) {
    const uint32_t query_pop = popcount(query, w64);
    for_each_intersection(rows, n, w64, query, [&](size_t i, uint32_t inter) {
        out[i] = jt_sim(inter, pops[i], query_pop);
    });
}

// Index of the first minimum of n values
inline size_t first_min(const double* v, size_t n) {
    return static_cast<size_t>(std::min_element(v, v + n) - v);
}

// iSIM Tanimoto of n_objects fingerprints, from sum_kq and sum_kqsq, the sum
// and the sum of squares of their column-wise sum (n_objects must be >= 2)
inline double isim(uint64_t sum_kq, uint64_t sum_kqsq, int64_t n_objects) {
    if (sum_kq == 0) {
        return 1.0;
    }
    auto a = (sum_kqsq - sum_kq) / 2.0;
    return a / ((a + (n_objects * sum_kq)) - sum_kqsq);
}

// Majority vote centroid value of a column with sum linear_sum over n_samples
// fingerprints
template <typename T>
inline uint8_t centroid_value(T linear_sum, int64_t n_samples) {
    if (n_samples <= 1) {
        // Narrowing conversion: if n_samples <= 1 then linear_sum is guaranteed
        // to have a value that a uint8_t can hold (it should be 0 or 1)
        return static_cast<uint8_t>(linear_sum);
    }
    return (linear_sum >= n_samples * 0.5) ? 1 : 0;
}

template <typename T>
inline bool centroid_bit(T linear_sum, int64_t n_samples) {
    return centroid_value(linear_sum, n_samples) != 0;
}

// Most dissimilar pair of n rows (O(N) approximation): fp1 is the first row
// least similar to the centroid of the rows, fp2 the first row least similar to
// fp1. The centroid uses the first n_columns bits only (the first n_columns / 8
// bytes of the packed fingerprints). Writes the similarities of all rows to fp1
// and fp2.
struct DissimilarPair {
    size_t fp1, fp2;
};

inline DissimilarPair most_dissimilar(const uint64_t* rows,
                                      const uint32_t* pops, size_t n,
                                      size_t w64, size_t n_columns,
                                      double* sims1, double* sims2) {
    std::vector<uint64_t> counts(w64 * 64, 0);
    for (size_t i = 0; i < n; ++i) {
        for_each_bit(rows + i * w64, w64, [&](int p) { ++counts[p]; });
    }
    std::vector<uint64_t> centroid(w64, 0);
    for (size_t p = 0; p < n_columns; ++p) {
        if (centroid_bit(counts[p], static_cast<int64_t>(n))) {
            centroid[p >> 6] |= uint64_t{1} << (p & 63);
        }
    }
    jt_sims(rows, pops, n, w64, centroid.data(), sims1);
    const size_t fp1 = first_min(sims1, n);
    jt_sims(rows, pops, n, w64, rows + fp1 * w64, sims1);
    const size_t fp2 = first_min(sims1, n);
    jt_sims(rows, pops, n, w64, rows + fp2 * w64, sims2);
    return {fp1, fp2};
}

}  // namespace bblean
