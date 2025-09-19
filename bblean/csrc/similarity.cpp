#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

// Scalar popcount intrinsics:
#if defined(__SSE_4_2__) || defined(_M_SSE4_2)
    // Compiler-portable, but *not available in systems that do not have SSE*
    // (which should be almost no CPUs nowadays)
    // Not actually vector instructions, they just live in the SSE header
    // Should be *exactly as fast* as __(builtin_)popcnt(ll) (compile to the same code)
    // 
    // nmmintrin.h is the SSE4.2 intrinsics (only) header for all compilers
    // NOTE: This ifdef is probably overkill, almost all cases should be covered by the
    // GCC|Clang|MSVC ifdefs, but it doesn't hurt to add it
#    include <nmmintrin.h>
#    define POPCOUNT_32 _mm_popcnt_u32
#    define POPCOUNT_64 _mm_popcnt_u64
#elif defined(_MSC_VER)
    // Windows (MSVC compiler)
#    include <intrin.h>
#    define POPCOUNT_32 __popcnt
#    define POPCOUNT_64 __popcnt64
#elif defined(__GNUC__) || defined(__clang__)
    // GCC | Clang
#   define POPCOUNT_32 __builtin_popcount
#   define POPCOUNT_64 __builtin_popcountll
#else
    // If popcnt is not hardware supported numpy rolls out its own hand-coded version,
    // fail for simplicity since it is not worth it to support those archs
#   error "Popcount not supported in target architecture"
#endif


// Vector popcount intrinsics:

// AVX(2) have no popcount intrinsics (_mm256_*)

// AVX-512 has vectorized popcount intrinsics (epi16 and epi32) (_mm512_*)
// NOTE: Ryzen 5 does not support AVX-512, it is only supported on very good CPU
// (only for CPU supporting it)
// Should be even faster
// TODO: Benchmark if this is worth it on HPC CPUs
// requires -mavx512* (* = smth, check what it is) in gcc (I think ?)
// (code compiled with -mavx* it will not run on older CPU, since GCC uses a different
// "instruction encoding" called VEX
// GCC: use __attribute__((target("avx512"))) or "ifunc" (I think ?)
// MSVC: use __cpuid (?)
// MSVC enables this by default (?) maybe needs /arch:AVX512 (?)

// #include <immintrin.h>  // All SIMD intrinsics, SSE, AVX, AVX2, AVX-512
// #define POPCOUNT_8_AVX _mm512_popcnt_epi8
// #define POPCOUNT_16_AVC _mm512_popcnt_epi16


// NOTE: For supporting multiple archs with different intrinsics availability, ifunc
// (GCC|Clang) may be useful, which allows conditional on-the-fly dispatch example:
// #include <immintrin.h>
// #include <cpuid.h>

// int add_sse(int a, int b) { return a + b; } // dummy example
// int add_avx(int a, int b) { return a + b; }

// static int (*resolve_add(void))(int,int) {
    // unsigned int eax, ebx, ecx, edx;
    // __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    // if (ecx & bit_AVX) return add_avx;
    // return add_sse;
// }

// int add(int a, int b) __attribute__((ifunc("resolve_add")));

// NOTE: MSVC uses <intrin.h>, __cpuid / __cpuidex, for cpuid, but has no support for ifunc

namespace py = pybind11;

// TODO: we can should be able to assume the input array size is always a multiple of
// 64, otherwise the hand-coded C++ code-path should not be triggered (?)

// TODO: Checking for alignment in the popcount fns should be done with:
//
// if (reinterpret_cast<uintptr_t>(row_ptr) % alignof(uint64_t) == 0)) {
// <reinterpret ptr to Ptr[uint64_t], loop over 8-byte chunks and call POPCOUNT_64>
// } else { <loop over the 1-byte elements, use POPCOUNT_32, which promotes to uint32> }
uint32_t _popcount_1d(const py::array_t<uint8_t>& arr) {

    // Input buffer
    const py::buffer_info in_bufinfo = arr.request();
    if (in_bufinfo.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    const py::ssize_t n_bytes = in_bufinfo.shape[0];
    auto in_ptr = static_cast<const uint8_t*>(in_bufinfo.ptr);

    // Output scalar
    uint32_t count{0};

    for (py::ssize_t i = 0; i < n_bytes; ++i) {
        // uint8 is promoted to uint32
        count += POPCOUNT_32(in_ptr[i]);
    }
    return count;
}

uint32_t _popcount_1d_reinterpret(const py::array_t<uint8_t>& arr) {

    // Input buffer
    const py::buffer_info in_bufinfo = arr.request();
    if (in_bufinfo.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    const py::ssize_t n_8bytes = in_bufinfo.shape[0] / 8;
    // TODO: Currently not safe, it is much faster for sure, but
    // should first check if the buffer is aligned to 8-bytes, otherwise unaligned
    // access may make this *super slow* in some cases in x64, and crash in aarch64
    // numpy arrays allocated as uint8_t are *not guaranteed to be aligned to 8-bytes*
    auto in_ptr_u64 = reinterpret_cast<const uint64_t*>(in_bufinfo.ptr);

    // Output scalar
    uint32_t count{0};

    for (py::ssize_t i = 0; i < n_8bytes; ++i) {
        count += POPCOUNT_64(in_ptr_u64[i]);
    }
    return count;
}

py::array_t<uint32_t> _popcount_2d_reinterpret(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& arr) {

    // Input bufffer
    py::buffer_info in_bufinfo = arr.request();
    if (in_bufinfo.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    auto in_ptr = static_cast<const uint8_t*>(in_bufinfo.ptr);
    const py::ssize_t n_samples = in_bufinfo.shape[0];
    const py::ssize_t n_8bytes = in_bufinfo.shape[1] / 8;

    // Output bufffer
    auto out = py::array_t<uint32_t>(n_samples);
    py::buffer_info out_bufinfo = out.request();
    auto out_ptr = static_cast<uint32_t*>(out_bufinfo.ptr);

    // TODO: Currently not safe, it is much faster for sure, but
    // should first check if the buffer is aligned to 8-bytes, otherwise unaligned
    // access may make this *super slow* in some cases in x64, and crash in aarch64
    // numpy arrays allocated as uint8_t are *not guaranteed to be aligned to 8-bytes*
    for (py::ssize_t i = 0; i < n_samples; ++i) {
        out_ptr[i] = 0;
        auto in_row_ptr_u64 = reinterpret_cast<const uint64_t*>(in_ptr + i * in_bufinfo.strides[0]);
        for (py::ssize_t j = 0; j < n_8bytes; ++j) {
            out_ptr[i] += POPCOUNT_64(in_row_ptr_u64[j]);
        }
    }
    return out;
}

// TODO: Currently this is pretty slow, maybe two pass approach? first compute all
// popcounts, then sum (Numpy does this). Maybe the additions could be auto-vectorized
// in that case.
py::array_t<uint32_t> _popcount_2d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a) {
    py::buffer_info a_buf = a.request();
    if (a_buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    const py::ssize_t n_samples = a_buf.shape[0];
    const py::ssize_t n_bytes = a_buf.shape[1];

    auto result = py::array_t<uint32_t>(n_samples);
    py::buffer_info result_buf = result.request();
    auto result_ptr = static_cast<uint32_t*>(result_buf.ptr);
    auto a_ptr = static_cast<const uint8_t*>(a_buf.ptr);

    for (py::ssize_t i = 0; i < n_samples; ++i) {
        result_ptr[i] = 0;
        const uint8_t* row_ptr = a_ptr + i * a_buf.strides[0];
        for (py::ssize_t j = 0; j < n_bytes; ++j) {
            // uint8 is promoted to uint32
            result_ptr[i] += POPCOUNT_32(row_ptr[j]);
        }
    }
    return result;
}
// The following is an attempt to explain numpy's "UNARY LOOPS"
// I believe this makes it "simpler for the compiler to optimize"
// I don't think numpy uses SIMD for the popcnt
//
// char *ip1 = args[0], *op1 = args[1];
// uint64_t is1 = steps[0], os1 = steps[1];
// uint64_t n = dimensions[0];
// uint64_t i;
// At the end of each iteration you increase both is1 and op1
// both the input pointer and the output pointer are Ptr[char]
// They have to be cast to the appropriate type, for example (uint64_t*)
// and dereferenced afterwards
//
// is1 = stride
// os1 = ?
// (0 increase the counter)
// (1 advance the input pointer, ip1, by a given stride)
// (2 advance the output pointer, op1 by a given stride)
// for(i = 0; i < n; i++, ip1 += is1, op1 += os1) {
// // Cast both the input and the output pointers,
// // the input pointer is derefereced to obtain the value. The value
// // is "const"
// const tin in = *(uint64_t *)ip1;
// tout *out = (tout *)op1;
// op;
// }

// char *ip1 = args[0], *op1 = args[1];
// npy_intp is1 = steps[0], os1 = steps[1];
// npy_intp n = dimensions[0];
// npy_intp i;
// for(i = 0; i < n; i++, ip1 += is1, op1 += os1) {
// const tin in = *(tin *)ip1;
// tout *out = (tout *)op1;
// op;
// }


// TODO: This works but it is *extremely slow*
py::array_t<uint8_t> _unpack_fingerprints_2d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a,
    std::optional<py::ssize_t> n_features_opt) {
    py::buffer_info a_buf = a.request();
    if (a_buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    py::ssize_t n_samples = a_buf.shape[0];
    py::ssize_t n_bytes = a_buf.shape[1];

    py::ssize_t n_features = n_features_opt.value_or(n_bytes * 8);

    auto result = py::array_t<uint8_t>({n_samples, n_features});
    py::buffer_info res_buf = result.request();

    auto a_ptr = static_cast<const uint8_t*>(a_buf.ptr);
    auto res_ptr = static_cast<uint8_t*>(res_buf.ptr);

    for (py::ssize_t i = 0; i < n_samples; ++i) {
        const uint8_t* row_ptr = a_ptr + i * a_buf.strides[0];
        for (py::ssize_t j = 0; j < n_bytes; ++j) {
            uint8_t byte = row_ptr[j];
            for (py::ssize_t k = 0; k < 8; ++k) {
                py::ssize_t feature_idx = j * 8 + k;
                if (feature_idx < n_features) {
                    res_ptr[i * res_buf.strides[0] + feature_idx] =
                        (byte >> (7 - k)) & 1;
                }
            }
        }
    }
    return result;
}

py::array_t<uint8_t> _calc_centroid_packed_u8_from_u64(
    const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>&
        linear_sum,
    int64_t n_samples) {
    py::buffer_info sum_buf = linear_sum.request();
    if (sum_buf.ndim != 1) {
        throw std::runtime_error("linear_sum must be 1-dimensional");
    }

    py::ssize_t n_features = sum_buf.shape[0];
    auto sum_ptr = static_cast<const uint64_t*>(sum_buf.ptr);

    std::vector<uint8_t> centroid_unpacked(n_features);

    if (n_samples <= 1) {
        for (py::ssize_t i = 0; i < n_features; ++i) {
            centroid_unpacked[i] = static_cast<uint8_t>(sum_ptr[i]);
        }
    } else {
        auto threshold = static_cast<double>(n_samples) * 0.5;
        for (py::ssize_t i = 0; i < n_features; ++i) {
            centroid_unpacked[i] =
                (static_cast<double>(sum_ptr[i]) >= threshold) ? 1 : 0;
        }
    }

    py::ssize_t n_bytes = (n_features + 7) / 8;
    auto centroid_packed = py::array_t<uint8_t>(n_bytes);
    py::buffer_info cent_packed_buf = centroid_packed.request();
    auto cent_packed_ptr = static_cast<uint8_t*>(cent_packed_buf.ptr);
    std::memset(cent_packed_ptr, 0, n_bytes);

    for (py::ssize_t i = 0; i < n_features; ++i) {
        if (centroid_unpacked[i] == 1) {
            cent_packed_ptr[i / 8] |= (1 << (7 - (i % 8)));
        }
    }

    return centroid_packed;
}

double jt_isim(const py::array_t<uint64_t, py::array::c_style |
                                               py::array::forcecast>& c_total,
               int64_t n_objects) {
    if (n_objects < 2) {
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "Invalid n_objects in isim. Expected n_objects >= 2", 1);
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Input buffer
    py::buffer_info in_bufinfo = c_total.request();
    if (in_bufinfo.ndim != 1) {
        throw std::runtime_error("c_total must be a 1D array");
    }
    py::ssize_t size = in_bufinfo.shape[0];
    auto in_ptr = static_cast<const uint64_t*>(in_bufinfo.ptr);

    uint64_t sum_kq{0};
    for (py::ssize_t i = 0; i < size; ++i) {
        sum_kq += in_ptr[i];
    }

    if (sum_kq == 0) {
        return 1.0;
    }

    uint64_t sum_kqsq{0};
    for (py::ssize_t i = 0; i < size; ++i) {
        sum_kqsq += in_ptr[i] * in_ptr[i];
    }
    auto a = static_cast<double>(sum_kqsq - sum_kq) / 2.0;
    return a / ((a + static_cast<double>(n_objects * sum_kq)) -
                static_cast<double>(sum_kqsq));
}

py::array_t<double> jt_sim_packed(
    py::array_t<uint8_t> arr, py::array_t<uint8_t> vec,
    std::optional<py::array_t<uint32_t>> cardinalities_opt) {
    py::buffer_info arr_buf = arr.request();
    py::buffer_info vec_buf = vec.request();

    if (arr_buf.ndim != 2 || vec_buf.ndim != 1) {
        throw std::runtime_error("arr must be 2D and vec must be 1D");
    }
    if (arr_buf.shape[1] != vec_buf.shape[0]) {
        throw std::runtime_error("arr and vec have incompatible shapes");
    }

    py::ssize_t n_samples = arr_buf.shape[0];
    py::ssize_t n_bytes = arr_buf.shape[1];

    py::array_t<uint32_t> cardinalities;
    if (cardinalities_opt) {
        cardinalities = cardinalities_opt.value();
    } else {
        cardinalities = _popcount_2d(arr);
    }
    py::buffer_info card_buf = cardinalities.request();
    auto card_ptr = static_cast<const uint32_t*>(card_buf.ptr);

    uint32_t vec_popcount = _popcount_1d(vec);

    auto result = py::array_t<double>(n_samples);
    py::buffer_info res_buf = result.request();
    auto res_ptr = static_cast<double*>(res_buf.ptr);
    auto arr_ptr = static_cast<const uint8_t*>(arr_buf.ptr);
    auto vec_ptr = static_cast<const uint8_t*>(vec_buf.ptr);

    for (py::ssize_t i = 0; i < n_samples; ++i) {
        const uint8_t* arr_row_ptr = arr_ptr + i * arr_buf.strides[0];
        uint32_t intersection{0};

        for (py::ssize_t j = 0; j < n_bytes; ++j) {
            // uint8 is promoted to uint32
            intersection += POPCOUNT_32(arr_row_ptr[j] & vec_ptr[j]);
        }
        auto denominator =
            static_cast<double>(card_ptr[i] + vec_popcount - intersection);
        res_ptr[i] =
            static_cast<double>(intersection) / std::max(denominator, 1.0);
    }

    return result;
}

py::tuple jt_most_dissimilar_packed(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> Y,
    std::optional<py::ssize_t> n_features_opt) {
    py::buffer_info Y_buf = Y.request();
    if (Y_buf.ndim != 2) {
        throw std::runtime_error("Y must be a 2D array");
    }
    py::ssize_t n_samples = Y_buf.shape[0];
    py::ssize_t n_features_packed = Y_buf.shape[1];
    auto Y_ptr = static_cast<const uint8_t*>(Y_buf.ptr);

    py::array_t<uint8_t> Y_unpacked =
        _unpack_fingerprints_2d(Y, n_features_opt);
    py::buffer_info Y_unpacked_buf = Y_unpacked.request();
    py::ssize_t n_features = Y_unpacked_buf.shape[1];
    auto Y_unpacked_ptr =
        static_cast<const uint8_t*>(Y_unpacked_buf.ptr);

    auto linear_sum = py::array_t<uint64_t>(n_features);
    py::buffer_info linear_sum_buf = linear_sum.request();
    auto linear_sum_ptr = static_cast<uint64_t*>(linear_sum_buf.ptr);
    std::fill(linear_sum_ptr, linear_sum_ptr + n_features, 0);

    for (py::ssize_t i = 0; i < n_samples; ++i) {
        const uint8_t* row_ptr = Y_unpacked_ptr + i * Y_unpacked_buf.strides[0];
        for (py::ssize_t j = 0; j < n_features; ++j) {
            linear_sum_ptr[j] += row_ptr[j];
        }
    }

    auto packed_centroid =
        _calc_centroid_packed_u8_from_u64(linear_sum, n_samples);
    auto cardinalities = _popcount_2d(Y);

    auto sims_cent = jt_sim_packed(Y, packed_centroid, cardinalities);
    py::buffer_info sims_cent_buf = sims_cent.request();
    auto sims_cent_ptr = static_cast<const double*>(sims_cent_buf.ptr);

    py::ssize_t fp_1_idx = std::distance(
        sims_cent_ptr,
        std::min_element(sims_cent_ptr, sims_cent_ptr + n_samples));

    auto fp_1_packed = py::array_t<uint8_t>(
        n_features_packed, Y_ptr + fp_1_idx * Y_buf.strides[0]);

    auto sims_fp_1 = jt_sim_packed(Y, fp_1_packed, cardinalities);
    py::buffer_info sims_fp_1_buf = sims_fp_1.request();
    auto sims_fp_1_ptr = static_cast<const double*>(sims_fp_1_buf.ptr);

    py::ssize_t fp_2_idx = std::distance(
        sims_fp_1_ptr,
        std::min_element(sims_fp_1_ptr, sims_fp_1_ptr + n_samples));

    auto fp_2_packed = py::array_t<uint8_t>(
        n_features_packed, Y_ptr + fp_2_idx * Y_buf.strides[0]);
    auto sims_fp_2 = jt_sim_packed(Y, fp_2_packed, cardinalities);

    return py::make_tuple(fp_1_idx, fp_2_idx, sims_fp_1, sims_fp_2);
}

PYBIND11_MODULE(_cpp_similarity, m) {
    m.doc() = "Optimized molecular similarity calculators (C++ extensions)";

    m.def("_unpack_fingerprints_2d", &_unpack_fingerprints_2d,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);

    m.def("_calc_centroid_packed_u8_from_u64",
          &_calc_centroid_packed_u8_from_u64, "Packed centroid calculation",
          py::arg("linear_sum"), py::arg("n_samples"));

    m.def("_popcount_2d_reinterpret", &_popcount_2d, "2D popcount", py::arg("a"));
    m.def("_popcount_1d_reinterpret", &_popcount_1d, "1D popcount", py::arg("a"));
    m.def("_popcount_2d", &_popcount_2d, "2D popcount", py::arg("a"));
    m.def("_popcount_1d", &_popcount_1d, "1D popcount", py::arg("a"));

    m.def("jt_isim", &jt_isim, "iSIM Tanimoto calculation", py::arg("c_total"),
          py::arg("n_objects"));

    m.def("jt_sim_packed", &jt_sim_packed,
          "Tanimoto similarity between a matrix of packed fps and a single "
          "packed fp",
          py::arg("arr"), py::arg("vec"),
          py::arg("_cardinalities") = std::nullopt);

    m.def("jt_most_dissimilar_packed", &jt_most_dissimilar_packed,
          "Finds two fps in a packed fp array that are the most "
          "Tanimoto-dissimilar",
          py::arg("Y"), py::arg("n_features") = std::nullopt);
}
