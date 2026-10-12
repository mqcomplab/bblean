#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

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

// Vector (SIMD) popcount kernels, selected at *compile time* from the flags
// the extension is built with. If no flag enables them (e.g. default build,
// MSVC, ARM / Apple Silicon) the portable scalar kernel is used instead:
// - AVX-512: needs AVX512F + AVX512BW + AVX512VPOPCNTDQ (Ice Lake+, Zen 4+)
//   (e.g. -march=native in a supporting CPU, or -mavx512f -mavx512bw
//   -mavx512vpopcntdq)
// - AVX2: Mula's nibble-lookup popcount (Haswell+, Zen+) (e.g. -mavx2)
// NOTE: An extension compiled with these flags will crash with "illegal
// instruction" in CPUs that don't support them
// TODO: Add NEON kernel (vcntq_u8) for ARM
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VPOPCNTDQ__)
#define BBLEAN_SIMD_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define BBLEAN_SIMD_AVX2 1
#include <immintrin.h>
#endif
#endif

namespace py = pybind11;

template <typename T>
using CArrayForcecast =
    py::array_t<T, py::array::c_style | py::array::forcecast>;

#if defined(BBLEAN_SIMD_AVX512)
constexpr const char* SIMD_BACKEND = "avx512";
#elif defined(BBLEAN_SIMD_AVX2)
constexpr const char* SIMD_BACKEND = "avx2";
#else
constexpr const char* SIMD_BACKEND = "scalar";
#endif

// All popcount kernels compute popcount(a) if AND is false, and
// popcount(a & b) if AND is true (b is ignored if AND is false). They work for
// any n_bytes and have no alignment requirements.

// Portable scalar kernel, processes 8 bytes at a time
template <bool AND>
inline uint64_t _popcount_scalar(const uint8_t* a, const uint8_t* b,
                                 py::ssize_t n_bytes) {
    uint64_t count{0};
    py::ssize_t i{0};
    for (; i + 8 <= n_bytes; i += 8) {
        // memcpy is the portable way of doing an unaligned load, it compiles
        // to a single mov
        uint64_t x;
        std::memcpy(&x, a + i, sizeof(uint64_t));
        if constexpr (AND) {
            uint64_t y;
            std::memcpy(&y, b + i, sizeof(uint64_t));
            x &= y;
        }
        count += POPCOUNT_64(x);
    }
    for (; i != n_bytes; ++i) {
        if constexpr (AND) {
            count += POPCOUNT_32(a[i] & b[i]);
        } else {
            count += POPCOUNT_32(a[i]);  // uint8 promoted to uint32
        }
    }
    return count;
}

#if defined(BBLEAN_SIMD_AVX512)
// Native 64-bit lane popcount (vpopcntq), 64 bytes per iteration. The tail is
// handled with a masked load (masked-out bytes are never read)
template <bool AND>
inline uint64_t _popcount_avx512(const uint8_t* a, const uint8_t* b,
                                 py::ssize_t n_bytes) {
    __m512i acc = _mm512_setzero_si512();
    py::ssize_t i{0};
    for (; i + 64 <= n_bytes; i += 64) {
        __m512i v = _mm512_loadu_si512(a + i);
        if constexpr (AND) {
            v = _mm512_and_si512(v, _mm512_loadu_si512(b + i));
        }
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    if (i != n_bytes) {
        const __mmask64 mask = ~uint64_t{0} >> (64 - (n_bytes - i));
        __m512i v = _mm512_maskz_loadu_epi8(mask, a + i);
        if constexpr (AND) {
            v = _mm512_and_si512(v, _mm512_maskz_loadu_epi8(mask, b + i));
        }
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    return _mm512_reduce_add_epi64(acc);
}
#endif

#if defined(BBLEAN_SIMD_AVX2)
// Mula's algorithm: each nibble's popcount is looked up in a 16-entry table
// with vpshufb, per-byte counts are accumulated in uint8 lanes, and
// periodically reduced to uint64 lanes with vpsadbw. 32 bytes per iteration,
// the tail (< 32 bytes) is handled by the scalar kernel
// See: Mula, Kurz, Lemire, "Faster Population Counts Using AVX2 Instructions"
template <bool AND>
inline uint64_t _popcount_avx2(const uint8_t* a, const uint8_t* b,
                               py::ssize_t n_bytes) {
    const __m256i lookup =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,  //
                         0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc = _mm256_setzero_si256();  // 4 x uint64 lanes
    py::ssize_t i{0};
    const py::ssize_t n_vec_bytes = n_bytes - n_bytes % 32;
    while (i != n_vec_bytes) {
        // Each iteration adds at most 8 to each uint8 lane, so at most 31
        // iterations (31 * 8 = 248) can be done before uint8 overflow
        const py::ssize_t block_end =
            std::min<py::ssize_t>(i + 31 * 32, n_vec_bytes);
        __m256i acc8 = _mm256_setzero_si256();  // 32 x uint8 lanes
        for (; i != block_end; i += 32) {
            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(a + i));
            if constexpr (AND) {
                v = _mm256_and_si256(v, _mm256_loadu_si256(
                                            reinterpret_cast<const __m256i*>(
                                                b + i)));
            }
            const __m256i lo = _mm256_and_si256(v, low_mask);
            const __m256i hi =
                _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            acc8 = _mm256_add_epi8(acc8, _mm256_shuffle_epi8(lookup, lo));
            acc8 = _mm256_add_epi8(acc8, _mm256_shuffle_epi8(lookup, hi));
        }
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(acc8, zero));
    }
    uint64_t count = static_cast<uint64_t>(_mm256_extract_epi64(acc, 0)) +
                     static_cast<uint64_t>(_mm256_extract_epi64(acc, 1)) +
                     static_cast<uint64_t>(_mm256_extract_epi64(acc, 2)) +
                     static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));
    return count +
           _popcount_scalar<AND>(a + i, AND ? b + i : b, n_bytes - i);
}
#endif

// Dispatch to the best kernel available at compile time
template <bool AND>
inline uint64_t _popcount_bytes(const uint8_t* a, const uint8_t* b,
                                py::ssize_t n_bytes) {
#if defined(BBLEAN_SIMD_AVX512)
    return _popcount_avx512<AND>(a, b, n_bytes);
#elif defined(BBLEAN_SIMD_AVX2)
    return _popcount_avx2<AND>(a, b, n_bytes);
#else
    return _popcount_scalar<AND>(a, b, n_bytes);
#endif
}

inline uint64_t popcount_bytes(const uint8_t* a, py::ssize_t n_bytes) {
    return _popcount_bytes<false>(a, nullptr, n_bytes);
}

inline uint64_t popcount_and_bytes(const uint8_t* a, const uint8_t* b,
                                   py::ssize_t n_bytes) {
    return _popcount_bytes<true>(a, b, n_bytes);
}

// Fused kernels: compute popcount(a) and popcount(a & b) in a single pass,
// loading each byte of a and b only once
struct CardAndIntersection {
    uint64_t card{0};
    uint64_t intersection{0};
};

inline CardAndIntersection _card_and_intersection_scalar(const uint8_t* a,
                                                         const uint8_t* b,
                                                         py::ssize_t n_bytes) {
    CardAndIntersection out;
    py::ssize_t i{0};
    for (; i + 8 <= n_bytes; i += 8) {
        uint64_t x, y;
        std::memcpy(&x, a + i, sizeof(uint64_t));
        std::memcpy(&y, b + i, sizeof(uint64_t));
        out.card += POPCOUNT_64(x);
        out.intersection += POPCOUNT_64(x & y);
    }
    for (; i != n_bytes; ++i) {
        out.card += POPCOUNT_32(a[i]);
        out.intersection += POPCOUNT_32(a[i] & b[i]);
    }
    return out;
}

#if defined(BBLEAN_SIMD_AVX512)
inline CardAndIntersection _card_and_intersection_avx512(const uint8_t* a,
                                                         const uint8_t* b,
                                                         py::ssize_t n_bytes) {
    __m512i acc_card = _mm512_setzero_si512();
    __m512i acc_inter = _mm512_setzero_si512();
    py::ssize_t i{0};
    for (; i + 64 <= n_bytes; i += 64) {
        const __m512i v = _mm512_loadu_si512(a + i);
        const __m512i w = _mm512_loadu_si512(b + i);
        acc_card = _mm512_add_epi64(acc_card, _mm512_popcnt_epi64(v));
        acc_inter = _mm512_add_epi64(
            acc_inter, _mm512_popcnt_epi64(_mm512_and_si512(v, w)));
    }
    if (i != n_bytes) {
        const __mmask64 mask = ~uint64_t{0} >> (64 - (n_bytes - i));
        const __m512i v = _mm512_maskz_loadu_epi8(mask, a + i);
        const __m512i w = _mm512_maskz_loadu_epi8(mask, b + i);
        acc_card = _mm512_add_epi64(acc_card, _mm512_popcnt_epi64(v));
        acc_inter = _mm512_add_epi64(
            acc_inter, _mm512_popcnt_epi64(_mm512_and_si512(v, w)));
    }
    return {static_cast<uint64_t>(_mm512_reduce_add_epi64(acc_card)),
            static_cast<uint64_t>(_mm512_reduce_add_epi64(acc_inter))};
}
#endif

#if defined(BBLEAN_SIMD_AVX2)
inline uint64_t _hsum_epi64_avx2(__m256i v) {
    return static_cast<uint64_t>(_mm256_extract_epi64(v, 0)) +
           static_cast<uint64_t>(_mm256_extract_epi64(v, 1)) +
           static_cast<uint64_t>(_mm256_extract_epi64(v, 2)) +
           static_cast<uint64_t>(_mm256_extract_epi64(v, 3));
}

// Same algorithm as _popcount_avx2, with one accumulator per count
inline CardAndIntersection _card_and_intersection_avx2(const uint8_t* a,
                                                       const uint8_t* b,
                                                       py::ssize_t n_bytes) {
    const __m256i lookup =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,  //
                         0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i zero = _mm256_setzero_si256();
    auto byte_popcounts = [&](__m256i v) {
        const __m256i lo = _mm256_and_si256(v, low_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                               _mm256_shuffle_epi8(lookup, hi));
    };
    __m256i acc_card = _mm256_setzero_si256();   // 4 x uint64 lanes
    __m256i acc_inter = _mm256_setzero_si256();  // 4 x uint64 lanes
    py::ssize_t i{0};
    const py::ssize_t n_vec_bytes = n_bytes - n_bytes % 32;
    while (i != n_vec_bytes) {
        // See _popcount_avx2 for the reason of the 31 iteration blocks
        const py::ssize_t block_end =
            std::min<py::ssize_t>(i + 31 * 32, n_vec_bytes);
        __m256i acc8_card = _mm256_setzero_si256();   // 32 x uint8 lanes
        __m256i acc8_inter = _mm256_setzero_si256();  // 32 x uint8 lanes
        for (; i != block_end; i += 32) {
            const __m256i v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i w =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            acc8_card = _mm256_add_epi8(acc8_card, byte_popcounts(v));
            acc8_inter = _mm256_add_epi8(acc8_inter,
                                         byte_popcounts(_mm256_and_si256(v, w)));
        }
        acc_card = _mm256_add_epi64(acc_card, _mm256_sad_epu8(acc8_card, zero));
        acc_inter =
            _mm256_add_epi64(acc_inter, _mm256_sad_epu8(acc8_inter, zero));
    }
    auto tail = _card_and_intersection_scalar(a + i, b + i, n_bytes - i);
    return {_hsum_epi64_avx2(acc_card) + tail.card,
            _hsum_epi64_avx2(acc_inter) + tail.intersection};
}
#endif

inline CardAndIntersection card_and_intersection(const uint8_t* a,
                                                 const uint8_t* b,
                                                 py::ssize_t n_bytes) {
#if defined(BBLEAN_SIMD_AVX512)
    return _card_and_intersection_avx512(a, b, n_bytes);
#elif defined(BBLEAN_SIMD_AVX2)
    return _card_and_intersection_avx2(a, b, n_bytes);
#else
    return _card_and_intersection_scalar(a, b, n_bytes);
#endif
}

uint32_t _popcount_1d(const CArrayForcecast<uint8_t>& arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
#ifdef DEBUG_LOGS
    py::print("DEBUG: _popcount_1d fn triggered simd backend:", SIMD_BACKEND);
#endif
    return static_cast<uint32_t>(popcount_bytes(arr.data(), arr.shape(0)));
}

py::array_t<uint32_t> _popcount_2d(const CArrayForcecast<uint8_t>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
#ifdef DEBUG_LOGS
    py::print("DEBUG: _popcount_2d fn triggered simd backend:", SIMD_BACKEND);
#endif
    const py::ssize_t n_samples = arr.shape(0);
    const py::ssize_t n_bytes = arr.shape(1);
    auto out = py::array_t<uint32_t>(n_samples);
    auto out_ptr = out.mutable_data();
    auto in_cptr = arr.data();
    for (py::ssize_t i{0}; i != n_samples; ++i) {
        out_ptr[i] =
            static_cast<uint32_t>(popcount_bytes(in_cptr + i * n_bytes, n_bytes));
    }
    return out;
}

// The BitToByte table has shape (256, 8), and holds, for each
// value in the range 0-255, a row with the 8 associated bits as uint8_t values
constexpr std::array<std::array<uint8_t, 8>, 256> makeByteToBitsLookupTable() {
    std::array<std::array<uint8_t, 8>, 256> byteToBits{};
    for (int i{0}; i != 256; ++i) {
        for (int b{0}; b != 8; ++b) {
            // Shift right by b and, and fetch the least-significant-bit by
            // and'ng with 1 = 000...1
            byteToBits[i][7 - b] = (i >> b) & 1;
        }
    }
    return byteToBits;
}

constexpr auto BYTE_TO_BITS = makeByteToBitsLookupTable();

py::array_t<uint8_t> _nochecks_unpack_fingerprints_1d(
    const CArrayForcecast<uint8_t>& packed_fps,
    std::optional<py::ssize_t> n_features_opt = std::nullopt) {
    py::ssize_t n_bytes = packed_fps.shape(0);
    py::ssize_t n_features = n_features_opt.value_or(n_bytes * 8);
    if (n_features % 8 != 0) {
        throw std::runtime_error("Only n_features divisible by 8 is supported");
    }
    auto out = py::array_t<uint8_t>(n_features);
    auto out_ptr = out.mutable_data();
    auto in_cptr = packed_fps.data();
    for (py::ssize_t j{0}; j != n_features; j += 8) {  // not auto-vec by GCC
        // Copy the next 8 uint8 values in one go
        std::memcpy(out_ptr + j, BYTE_TO_BITS[in_cptr[j / 8]].data(), 8);
    }
    return out;
}

py::array_t<uint8_t> _nochecks_unpack_fingerprints_2d(
    const CArrayForcecast<uint8_t>& packed_fps,
    std::optional<py::ssize_t> n_features_opt = std::nullopt) {
    py::ssize_t n_samples = packed_fps.shape(0);
    py::ssize_t n_bytes = packed_fps.shape(1);
    py::ssize_t n_features = n_features_opt.value_or(n_bytes * 8);
    if (n_features % 8 != 0) {
        throw std::runtime_error("Only features divisible by 8 is supported");
    }
    auto out = py::array_t<uint8_t>({n_samples, n_features});
    // Unchecked accessors (benchmarked and there is no real advantage to using
    // ptrs)
    auto acc_in = packed_fps.unchecked<2>();
    auto acc_out = out.mutable_unchecked<2>();

    for (py::ssize_t i{0}; i != n_samples; ++i) {  // not auto-vec by GCC
        for (py::ssize_t j{0}; j != n_features;
             j += 8) {  // not auto-vec by GCC
            // Copy the next 8 uint8 values in one go
            std::memcpy(&acc_out(i, j), BYTE_TO_BITS[acc_in(i, j / 8)].data(),
                        8);
        }
    }
    return out;
}

// Wrapper over _nochecks_unpack_fingerprints that performs ndim checks
py::array_t<uint8_t> unpack_fingerprints(
    const CArrayForcecast<uint8_t>& packed_fps,
    std::optional<py::ssize_t> n_features_opt = std::nullopt) {
    if (packed_fps.ndim() == 1) {
        return _nochecks_unpack_fingerprints_1d(packed_fps, n_features_opt);
    }
    if (packed_fps.ndim() == 2) {
        return _nochecks_unpack_fingerprints_2d(packed_fps, n_features_opt);
    }
    throw std::runtime_error("Input array must be 1- or 2-dimensional");
}

template <typename T>
py::array_t<uint8_t> centroid_from_sum(const CArrayForcecast<T>& linear_sum,
                                       int64_t n_samples, bool pack = true) {
    if (linear_sum.ndim() != 1) {
        throw std::runtime_error("linear_sum must be 1-dimensional");
    }

    py::ssize_t n_features = linear_sum.shape(0);
    auto linear_sum_cptr = linear_sum.data();

    py::array_t<uint8_t> centroid_unpacked(n_features);
    auto centroid_unpacked_ptr = centroid_unpacked.mutable_data();
    if (n_samples <= 1) {
        for (int i{0}; i != n_features;
             ++i) {  // yes auto-vec by GCC (versioned due to possible alias)
            // Cast not required, but added for clarity since this is a
            // narrowing conversion. if n_samples <= 1 then linear_sum is
            // guaranteed to have a value that a uint8_t can hold (it should be
            // 0 or 1)
            // memcpy not possible due to the required cast
            centroid_unpacked_ptr[i] = static_cast<uint8_t>(linear_sum_cptr[i]);
        }
    } else {
        auto threshold = n_samples * 0.5;
        for (int i{0}; i != n_features; ++i) {  // not auto-vec by GCC
            centroid_unpacked_ptr[i] =
                (linear_sum_cptr[i] >= threshold) ? 1 : 0;
        }
    }

    if (!pack) {
        return centroid_unpacked;
    }

    auto centroid_unpacked_cptr = centroid_unpacked.data();
    int n_bytes = (n_features + 7) / 8;
    auto centroid_packed = py::array_t<uint8_t>(n_bytes);
    auto centroid_packed_ptr = centroid_packed.mutable_data();
    std::memset(centroid_packed_ptr, 0, centroid_packed.nbytes());

    // Slower than numpy, due to lack of SIMD
    // The following loop is *marginally slower* (benchmkd') than the
    // implemented one: for (int i{0}; i != n_features; ++i) {
    //    if (centroid_unpacked_cptr[i]) {
    //        centroid_packed_ptr[i / 8] |= (1 << (7 - (i % 8)));
    //    }
    //  }
    //  TODO: Check if GCC is auto-vectorizing
    for (int i{0}, stride{0}; i != n_bytes; i++, stride += 8) {
        for (int b{0}; b != 8; ++b) {
            centroid_packed_ptr[i] <<= 1;
            centroid_packed_ptr[i] |= centroid_unpacked_cptr[stride + b];
        }
    }
    return centroid_packed;
}

double jt_isim_from_sum(const CArrayForcecast<uint64_t>& linear_sum,
                        int64_t n_objects) {
    if (n_objects < 2) {
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "Invalid n_objects in isim. Expected n_objects >= 2", 1);
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (linear_sum.ndim() != 1) {
        throw std::runtime_error("linear_sum must be a 1D array");
    }
    py::ssize_t n_features = linear_sum.shape(0);

    auto in_cptr = linear_sum.data();
    uint64_t sum_kq{0};
    for (py::ssize_t i{0}; i != n_features; ++i) {  // yes auto-vec by GCC
        sum_kq += in_cptr[i];
    }

    if (sum_kq == 0) {
        return 1.0;
    }

    uint64_t sum_kqsq{0};
    for (py::ssize_t i{0}; i != n_features; ++i) {  // yes auto-vec by GCC
        sum_kqsq += in_cptr[i] * in_cptr[i];
    }
    auto a = (sum_kqsq - sum_kq) / 2.0;
    return a / ((a + (n_objects * sum_kq)) - sum_kqsq);
}

// NOTE: This is only *slightly* faster for C++ than numpy, **only if the
// array is uint8_t** if the array is uint64 already, it is slower
template <typename T>
py::array_t<uint64_t> add_rows(const CArrayForcecast<T>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    auto arr_ptr = arr.data();
    auto out = py::array_t<uint64_t>(arr.shape(1));
    auto out_ptr = out.mutable_data();
    std::memset(out_ptr, 0, out.nbytes());
    py::ssize_t n_samples = arr.shape(0);
    py::ssize_t n_features = arr.shape(1);
    // Check GCC / CLang vectorize this
    for (py::ssize_t i = 0; i < n_samples; ++i) {
        const uint8_t* arr_row_ptr = arr_ptr + i * n_features;
        for (py::ssize_t j = 0; j < n_features; ++j) {
            out_ptr[j] += static_cast<uint64_t>(arr_row_ptr[j]);
        }
    }
    return out;
}
py::array_t<double> _nochecks_jt_compl_isim_unpacked_u8(
    const py::array_t<uint8_t, py::array::c_style>& fps) {
    py::ssize_t n_objects = fps.shape(0);
    py::ssize_t n_features = fps.shape(1);
    auto out = py::array_t<double>(n_objects);
    auto out_ptr = out.mutable_data();

    if (n_objects < 3) {
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "Invalid num fps in compl_isim. Expected n_objects >= 3",
                     1);
        for (py::ssize_t i{0}; i != n_objects; ++i) {
            out_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return out;
    }

    auto linear_sum = add_rows<uint8_t>(fps);
    auto ls_cptr = linear_sum.data();

    py::array_t<uint64_t> shifted_linear_sum(n_features);
    auto shifted_ls_ptr = shifted_linear_sum.mutable_data();

    auto in_cptr = fps.data();
    for (py::ssize_t i{0}; i != n_objects; ++i) {
        for (py::ssize_t j{0}; j != n_features; ++j) {
            shifted_ls_ptr[j] = ls_cptr[j] - in_cptr[i * n_features + j];
        }
        // For all compl isim N is n_objects - 1
        out_ptr[i] = jt_isim_from_sum(shifted_linear_sum, n_objects - 1);
    }
    return out;
}

py::array_t<double> jt_compl_isim(
    const CArrayForcecast<uint8_t>& fps, bool input_is_packed = true,
    std::optional<py::ssize_t> n_features_opt = std::nullopt) {
    if (fps.ndim() != 2) {
        throw std::runtime_error("fps arr must be 2D");
    }
    if (input_is_packed) {
        return _nochecks_jt_compl_isim_unpacked_u8(
            _nochecks_unpack_fingerprints_2d(fps, n_features_opt));
    }
    return _nochecks_jt_compl_isim_unpacked_u8(fps);
}

inline double _jt_from_counts(uint64_t intersection, uint64_t card_a,
                               uint64_t card_b) {
    auto denominator = card_a + card_b - intersection;
    // Cast is technically unnecessary since std::max promotes to double,
    // but added here for clarity (should compile to nop)
    return intersection / std::max(static_cast<double>(denominator), 1.0);
}

void _check_arr_vec_shapes(const CArrayForcecast<uint8_t>& arr,
                           const CArrayForcecast<uint8_t>& vec) {
    if (arr.ndim() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("arr must be 2D, vec must be 1D");
    }
    if (arr.shape(1) != vec.shape(0)) {
        throw std::runtime_error(
            "Shapes should be (N, F) for arr and (F,) for vec");
    }
}

// # NOTE: This function is the bottleneck for bb compute calculations
py::array_t<double> jt_sim_packed_precalc_cardinalities(
    const CArrayForcecast<uint8_t>& arr, const CArrayForcecast<uint8_t>& vec,
    const py::array_t<uint32_t>& cardinalities) {
    _check_arr_vec_shapes(arr, vec);
#ifdef DEBUG_LOGS
    py::print("DEBUG: jt_sim_packed fn triggered simd backend:", SIMD_BACKEND);
#endif
    const py::ssize_t n_samples = arr.shape(0);
    const py::ssize_t n_bytes = arr.shape(1);
    auto out = py::array_t<double>(n_samples);
    auto out_ptr = out.mutable_data();
    auto arr_cptr = arr.data();
    auto vec_cptr = vec.data();
    auto card_cptr = cardinalities.data();
    const uint64_t vec_popcount = popcount_bytes(vec_cptr, n_bytes);
    for (py::ssize_t i{0}; i != n_samples; ++i) {
        const uint64_t intersection =
            popcount_and_bytes(arr_cptr + i * n_bytes, vec_cptr, n_bytes);
        out_ptr[i] = _jt_from_counts(intersection, card_cptr[i], vec_popcount);
    }
    return out;
}

// Computes the cardinalities and intersections in a single pass over arr,
// instead of a separate pass to calculate the cardinalities
py::array_t<double> _jt_sim_arr_vec_packed(
    const CArrayForcecast<uint8_t>& arr, const CArrayForcecast<uint8_t>& vec) {
    _check_arr_vec_shapes(arr, vec);
#ifdef DEBUG_LOGS
    py::print("DEBUG: jt_sim_packed fn triggered simd backend:", SIMD_BACKEND);
#endif
    const py::ssize_t n_samples = arr.shape(0);
    const py::ssize_t n_bytes = arr.shape(1);
    auto out = py::array_t<double>(n_samples);
    auto out_ptr = out.mutable_data();
    auto arr_cptr = arr.data();
    auto vec_cptr = vec.data();
    const uint64_t vec_popcount = popcount_bytes(vec_cptr, n_bytes);
    for (py::ssize_t i{0}; i != n_samples; ++i) {
        const auto counts =
            card_and_intersection(arr_cptr + i * n_bytes, vec_cptr, n_bytes);
        out_ptr[i] =
            _jt_from_counts(counts.intersection, counts.card, vec_popcount);
    }
    return out;
}

double jt_isim_unpacked_u8(const CArrayForcecast<uint8_t>& arr) {
    return jt_isim_from_sum(add_rows<uint8_t>(arr), arr.shape(0));
}

double jt_isim_packed_u8(
    const CArrayForcecast<uint8_t>& arr,
    std::optional<py::ssize_t> n_features_opt = std::nullopt) {
    return jt_isim_from_sum(
        add_rows<uint8_t>(unpack_fingerprints(arr, n_features_opt)),
        arr.shape(0));
}

py::tuple jt_most_dissimilar_packed(
    CArrayForcecast<uint8_t> fps_packed,
    std::optional<py::ssize_t> n_features_opt = std::nullopt) {
    if (fps_packed.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    py::ssize_t n_samples = fps_packed.shape(0);
    py::ssize_t n_features_packed = fps_packed.shape(1);

    auto fps_unpacked =
        _nochecks_unpack_fingerprints_2d(fps_packed, n_features_opt);
    py::ssize_t n_features_unpacked = fps_unpacked.shape(1);

    auto linear_sum = py::array_t<uint64_t>(n_features_unpacked);
    auto linear_sum_ptr = linear_sum.mutable_data();
    std::memset(linear_sum_ptr, 0, linear_sum.nbytes());

    // TODO: This sum could be vectorized manually or automatically
    auto fps_unpacked_ptr = fps_unpacked.data();
    for (py::ssize_t i{0}; i != n_samples; ++i) {
        const uint8_t* row_cptr = fps_unpacked_ptr + i * n_features_unpacked;
        for (py::ssize_t j{0}; j != n_features_unpacked;
             ++j) {  // yes auto-vec by GCC (versioned due to possible alias)
            linear_sum_ptr[j] += row_cptr[j];
        }
    }

    auto centroid_packed =
        centroid_from_sum<uint64_t>(linear_sum, n_samples, true);
    auto cardinalities = _popcount_2d(fps_packed);

    auto sims_cent = jt_sim_packed_precalc_cardinalities(
        fps_packed, centroid_packed, cardinalities);
    auto sims_cent_ptr = sims_cent.data();

    auto fps_packed_cptr = fps_packed.data();

    // argmin
    py::ssize_t fp1_idx = std::distance(
        sims_cent_ptr,
        std::min_element(sims_cent_ptr, sims_cent_ptr + n_samples));
    auto fp1_packed = CArrayForcecast<uint8_t>(
        n_features_packed, fps_packed_cptr + fp1_idx * n_features_packed);

    auto sims_fp1 = jt_sim_packed_precalc_cardinalities(fps_packed, fp1_packed,
                                                        cardinalities);
    auto sims_fp1_ptr = sims_fp1.data();

    // argmin
    py::ssize_t fp2_idx = std::distance(
        sims_fp1_ptr, std::min_element(sims_fp1_ptr, sims_fp1_ptr + n_samples));
    auto fp2_packed = CArrayForcecast<uint8_t>(
        n_features_packed, fps_packed_cptr + fp2_idx * n_features_packed);

    auto sims_fp2 = jt_sim_packed_precalc_cardinalities(fps_packed, fp2_packed,
                                                        cardinalities);

    return py::make_tuple(fp1_idx, fp2_idx, sims_fp1, sims_fp2);
}

PYBIND11_MODULE(_cpp_similarity, m) {
    m.doc() = "Optimized molecular similarity calculators (C++ extensions)";
    // Popcount kernel selected at compile time ("avx512", "avx2" or "scalar")
    m.attr("_simd_backend") = SIMD_BACKEND;

    // Only bound for debugging purposes
    m.def("_nochecks_unpack_fingerprints_2d", &_nochecks_unpack_fingerprints_2d,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);
    m.def("_nochecks_unpack_fingerprints_1d", &_nochecks_unpack_fingerprints_1d,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);

    // NOTE: There are some gains from using this fn but only ~3%, so don't warn
    // for now if this fails, and don't expose it
    m.def("unpack_fingerprints", &unpack_fingerprints,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);

    // NOTE: pybind11's dynamic dispatch is *significantly* more
    // expensive than casting to uint64_t always
    // still this function is *barely* faster than python if no casts are
    // needed, and slightly slower if casts are needed so it is not useful
    // outside the C++ code, and it should not be exposed by default in any
    // module (only for internal use and debugging)
    m.def("centroid_from_sum", &centroid_from_sum<uint64_t>,
          "centroid calculation", py::arg("linear_sum"), py::arg("n_samples"),
          py::arg("pack") = true);

    m.def("_popcount_2d", &_popcount_2d, "2D popcount", py::arg("a"));
    m.def("_popcount_1d", &_popcount_1d, "1D popcount", py::arg("a"));
    m.def("add_rows", &add_rows<uint8_t>, "add_rows", py::arg("arr"));

    // API
    m.def("jt_isim_from_sum", &jt_isim_from_sum,
          "iSIM Tanimoto calculation from sum", py::arg("c_total"),
          py::arg("n_objects"));
    m.def("jt_isim_packed_u8", &jt_isim_packed_u8, "iSIM Tanimoto calculation",
          py::arg("arr"), py::arg("n_features") = std::nullopt);
    m.def("jt_isim_unpacked_u8", &jt_isim_unpacked_u8,
          "iSIM Tanimoto calculation", py::arg("arr"));

    m.def("jt_compl_isim", &jt_compl_isim, "Complementary iSIM tanimoto",
          py::arg("fps"), py::arg("input_is_packed") = true,
          py::arg("n_features") = std::nullopt);

    m.def("_jt_sim_arr_vec_packed", &_jt_sim_arr_vec_packed,
          "Tanimoto similarity between a matrix of packed fps and a single "
          "packed fp",
          py::arg("arr"), py::arg("vec"));
    m.def("jt_most_dissimilar_packed", &jt_most_dissimilar_packed,
          "Finds two fps in a packed fp array that are the most "
          "Tanimoto-dissimilar",
          py::arg("Y"), py::arg("n_features") = std::nullopt);
}
