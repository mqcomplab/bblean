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
#include <nmmintrin.h>

#ifdef _MSC_VER
// Windows (MSVC compiler)
#include <intrin.h>
#define POPCOUNT __popcnt
#define POPCOUNTLL __popcnt64
#else
// GCC | Clang
#define POPCOUNT __builtin_popcount
// #define POPCOUNTLL __builtin_popcountll
// This is significanlty faster uses SSE4.2
#define POPCOUNTLL _mm_popcnt_u64
// Even faster, uses AVX-512, requers -mavx512f -mavx512dq (?)
// #define POPCOUNTLL _mm512_popcnt_epi64
#endif

namespace py = pybind11;

namespace {  // Anonymous namespace for helper functions

// Unsafe popcount paths:
// Similarly, choose simple code path here
// if (n_bytes % 8 == 0 &&
// (reinterpret_cast<uintptr_t>(row_ptr) % alignof(uint64_t) == 0)) {
// const uint64_t* p64 = reinterpret_cast<const uint64_t*>(row_ptr);
// for (std::size_t j = 0; j < n_bytes / 8; ++j) {
// result_ptr[i] += POPCOUNTLL(p64[j]);
// }
// } else {
// }

// NOTE: we can Assume the input array size is always a multiple of 64,
// otherwise the C++ code-path should not be triggered For now, however,
// chose the "safe" codepath to ensure correct alignment
//
// Helper for popcount on a 1D uint8 array
uint32_t _popcount_1d(const py::array_t<uint8_t>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    const std::size_t n_bytes = buf.shape[0];
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    uint32_t count{0u};

    for (std::size_t i = 0; i < n_bytes; ++i) {
        count += POPCOUNT(ptr[i]);
    }
    return count;
}

// Corresponds to _popcount in Python
py::array_t<uint32_t> _popcount_2d(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a) {
    auto a_buf = a.request();
    if (a_buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    const py::ssize_t n_samples = a_buf.shape[0];
    const py::ssize_t n_8bytes = a_buf.shape[1] / 8;

    auto result = py::array_t<uint32_t>(n_samples);
    auto result_buf = result.request();
    uint32_t* result_ptr = static_cast<uint32_t*>(result_buf.ptr);
    const uint8_t* a_ptr = static_cast<const uint8_t*>(a_buf.ptr);

    // Two pass approach, first compute all popcounts, then sum
    // For the popcount computation, just go over all bytes of the array in order and
    // compute the popcount (maybe could be useful)
    // auto popcounts = py::array_t<uint8_t>(n_samples, n_features);


    for (py::ssize_t i = 0; i < n_samples; ++i) {
        result_ptr[i] = 0u;
        // TODO: is this safe??? it is much faster for sure
        // for (std::size_t j = 0; j < n_bytes; ++j) {
        const uint64_t* row_ptr_u64 = reinterpret_cast<const uint64_t*>(a_ptr + i * a_buf.strides[0]);
        for (py::ssize_t j = 0; j < n_8bytes; ++j) {
            result_ptr[i] += POPCOUNTLL(row_ptr_u64[j]);
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

py::array_t<uint8_t> _unpack_fingerprints_2d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a,
    std::optional<std::size_t> n_features_opt) {
    auto a_buf = a.request();
    if (a_buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    std::size_t n_samples = a_buf.shape[0];
    std::size_t n_bytes = a_buf.shape[1];

    std::size_t n_features = n_features_opt.value_or(n_bytes * 8);

    auto result = py::array_t<uint8_t>({n_samples, n_features});
    auto res_buf = result.request();

    const uint8_t* a_ptr = static_cast<const uint8_t*>(a_buf.ptr);
    uint8_t* res_ptr = static_cast<uint8_t*>(res_buf.ptr);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const uint8_t* row_ptr = a_ptr + i * a_buf.strides[0];
        for (std::size_t j = 0; j < n_bytes; ++j) {
            uint8_t byte = row_ptr[j];
            for (std::size_t k = 0; k < 8; ++k) {
                std::size_t feature_idx = j * 8 + k;
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
    auto sum_buf = linear_sum.request();
    if (sum_buf.ndim != 1) {
        throw std::runtime_error("linear_sum must be 1-dimensional");
    }

    std::size_t n_features = sum_buf.shape[0];
    const uint64_t* sum_ptr = static_cast<const uint64_t*>(sum_buf.ptr);

    std::vector<uint8_t> centroid_unpacked(n_features);

    if (n_samples <= 1) {
        for (std::size_t i = 0; i < n_features; ++i) {
            centroid_unpacked[i] = static_cast<uint8_t>(sum_ptr[i]);
        }
    } else {
        double threshold = static_cast<double>(n_samples) * 0.5;
        for (std::size_t i = 0; i < n_features; ++i) {
            centroid_unpacked[i] =
                (static_cast<double>(sum_ptr[i]) >= threshold) ? 1 : 0;
        }
    }

    std::size_t n_bytes = (n_features + 7) / 8;
    auto centroid_packed = py::array_t<uint8_t>(n_bytes);
    auto cent_packed_buf = centroid_packed.request();
    uint8_t* cent_packed_ptr = static_cast<uint8_t*>(cent_packed_buf.ptr);
    std::memset(cent_packed_ptr, 0, n_bytes);

    for (std::size_t i = 0; i < n_features; ++i) {
        if (centroid_unpacked[i] == 1) {
            cent_packed_ptr[i / 8] |= (1 << (7 - (i % 8)));
        }
    }

    return centroid_packed;
}
}  // namespace

double jt_isim(const py::array_t<uint64_t, py::array::c_style |
                                               py::array::forcecast>& c_total,
               int64_t n_objects) {
    if (n_objects < 2) {
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "Invalid n_objects in isim. Expected n_objects >= 2", 1);
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto buf = c_total.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("c_total must be a 1D array");
    }

    std::size_t size = buf.shape[0];
    const uint64_t* ptr = static_cast<const uint64_t*>(buf.ptr);

    uint64_t sum_kq{0};
    for (std::size_t i = 0; i < size; ++i) {
        sum_kq += ptr[i];
    }

    if (sum_kq == 0) {
        return 1.0;
    }

    uint64_t sum_kqsq{0};
    for (std::size_t i = 0; i < size; ++i) {
        sum_kqsq += ptr[i] * ptr[i];
    }
    const double a = static_cast<double>(sum_kqsq - sum_kq) / 2.0;
    return a / ((a + static_cast<double>(n_objects * sum_kq)) -
                static_cast<double>(sum_kqsq));
}

py::array_t<double> jt_sim_packed(
    py::array_t<uint8_t> arr, py::array_t<uint8_t> vec,
    std::optional<py::array_t<uint32_t>> cardinalities_opt) {
    auto arr_buf = arr.request();
    auto vec_buf = vec.request();

    if (arr_buf.ndim != 2 || vec_buf.ndim != 1) {
        throw std::runtime_error("arr must be 2D and vec must be 1D");
    }
    if (arr_buf.shape[1] != vec_buf.shape[0]) {
        throw std::runtime_error("arr and vec have incompatible shapes");
    }

    std::size_t n_samples = arr_buf.shape[0];
    std::size_t n_bytes = arr_buf.shape[1];

    py::array_t<uint32_t> cardinalities;
    if (cardinalities_opt) {
        cardinalities = cardinalities_opt.value();
    } else {
        cardinalities = _popcount_2d(arr);
    }
    auto card_buf = cardinalities.request();
    const uint32_t* card_ptr = static_cast<const uint32_t*>(card_buf.ptr);

    uint32_t vec_popcount = _popcount_1d(vec);

    auto result = py::array_t<double>(n_samples);
    auto res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);
    const uint8_t* arr_ptr = static_cast<const uint8_t*>(arr_buf.ptr);
    const uint8_t* vec_ptr = static_cast<const uint8_t*>(vec_buf.ptr);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const uint8_t* arr_row_ptr = arr_ptr + i * arr_buf.strides[0];
        uint32_t intersection{0u};

        for (std::size_t j = 0; j < n_bytes; ++j) {
            intersection += POPCOUNT(arr_row_ptr[j] & vec_ptr[j]);
        }
        double denominator =
            static_cast<double>(card_ptr[i] + vec_popcount - intersection);
        res_ptr[i] =
            static_cast<double>(intersection) / std::max(denominator, 1.0);
    }

    return result;
}

py::tuple jt_most_dissimilar_packed(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> Y,
    std::optional<std::size_t> n_features_opt) {
    auto Y_buf = Y.request();
    if (Y_buf.ndim != 2) {
        throw std::runtime_error("Y must be a 2D array");
    }
    std::size_t n_samples = Y_buf.shape[0];
    std::size_t n_features_packed = Y_buf.shape[1];
    const uint8_t* Y_ptr = static_cast<const uint8_t*>(Y_buf.ptr);

    py::array_t<uint8_t> Y_unpacked =
        _unpack_fingerprints_2d(Y, n_features_opt);
    auto Y_unpacked_buf = Y_unpacked.request();
    std::size_t n_features = Y_unpacked_buf.shape[1];
    const uint8_t* Y_unpacked_ptr =
        static_cast<const uint8_t*>(Y_unpacked_buf.ptr);

    auto linear_sum = py::array_t<uint64_t>(n_features);
    auto linear_sum_buf = linear_sum.request();
    uint64_t* linear_sum_ptr = static_cast<uint64_t*>(linear_sum_buf.ptr);
    std::fill(linear_sum_ptr, linear_sum_ptr + n_features, 0);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const uint8_t* row_ptr = Y_unpacked_ptr + i * Y_unpacked_buf.strides[0];
        for (std::size_t j = 0; j < n_features; ++j) {
            linear_sum_ptr[j] += row_ptr[j];
        }
    }

    auto packed_centroid =
        _calc_centroid_packed_u8_from_u64(linear_sum, n_samples);
    auto cardinalities = _popcount_2d(Y);

    auto sims_cent = jt_sim_packed(Y, packed_centroid, cardinalities);
    auto sims_cent_buf = sims_cent.request();
    const double* sims_cent_ptr = static_cast<const double*>(sims_cent_buf.ptr);

    std::size_t fp_1_idx = std::distance(
        sims_cent_ptr,
        std::min_element(sims_cent_ptr, sims_cent_ptr + n_samples));

    auto fp_1_packed = py::array_t<uint8_t>(
        n_features_packed, Y_ptr + fp_1_idx * Y_buf.strides[0]);

    auto sims_fp_1 = jt_sim_packed(Y, fp_1_packed, cardinalities);
    auto sims_fp_1_buf = sims_fp_1.request();
    const double* sims_fp_1_ptr = static_cast<const double*>(sims_fp_1_buf.ptr);

    std::size_t fp_2_idx = std::distance(
        sims_fp_1_ptr,
        std::min_element(sims_fp_1_ptr, sims_fp_1_ptr + n_samples));

    auto fp_2_packed = py::array_t<uint8_t>(
        n_features_packed, Y_ptr + fp_2_idx * Y_buf.strides[0]);
    auto sims_fp_2 = jt_sim_packed(Y, fp_2_packed, cardinalities);

    return py::make_tuple(fp_1_idx, fp_2_idx, sims_fp_1, sims_fp_2);
}

PYBIND11_MODULE(cpp_similarity, m) {
    m.doc() = "Optimized molecular similarity calculators (C++ extensions)";

    m.def("_unpack_fingerprints_2d", &_unpack_fingerprints_2d,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);

    m.def("_calc_centroid_packed_u8_from_u64",
          &_calc_centroid_packed_u8_from_u64, "Packed centroid calculation",
          py::arg("linear_sum"), py::arg("n_samples"));

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
