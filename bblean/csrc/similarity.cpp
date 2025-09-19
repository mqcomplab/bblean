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

// TODO: See if worth it to use vector popcount intrinsics (AVX-512, only some CPU)
// TODO: Refactor this code to use a templated function with uint8_t / uint64_t
// like jt_sim_packed
namespace py = pybind11;

// TODO: we can should be able to assume the input array size is always a
// multiple of 64, otherwise the hand-coded C++ code-path should not be
// triggered (?)
uint32_t _popcount_1d(const py::array_t<uint8_t>& arr) {
    // Input buffer set-up
    const py::buffer_info in_bufinfo = arr.request();
    if (in_bufinfo.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }

    // Output scalar set-up
    uint32_t count{0};

#ifdef DEBUG_LOGS
    std::cout << "in_buf_ptr addr: "
              << reinterpret_cast<std::uintptr_t>(in_bufinfo.ptr) << std::endl;
    std::cout << "uint64_t alignment requirement: " << alignof(uint64_t)
              << std::endl;
    std::cout << "Allignment check (coorect if 0): "
              << reinterpret_cast<std::uintptr_t>(in_bufinfo.ptr) %
                     alignof(uint64_t)
              << std::endl;
#endif
    // Conversion between const void* and const std::uintptr_t requires reinterpret
    if (reinterpret_cast<std::uintptr_t>(in_bufinfo.ptr) % alignof(uint64_t) ==
        0) {
#ifdef DEBUG_LOGS
        std::cout << "DEBUG: _popcount_1d triggered uint64 + popcount 64 branch"
                  << std::endl;
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        const py::ssize_t steps = in_bufinfo.shape[0] / 8;
        auto in_ptr = static_cast<const uint64_t*>(in_bufinfo.ptr);
        for (py::ssize_t i = 0; i < steps; ++i) {
            count += POPCOUNT_64(in_ptr[i]);
        }
    } else {
#ifdef DEBUG_LOGS
        std::cout << "DEBUG: _popcount_1d triggered uint8 + popcount 32 branch"
                  << std::endl;
#endif
        // Misaligned, loop over bytes
        const py::ssize_t steps = in_bufinfo.shape[0];
        auto in_ptr = static_cast<const uint8_t*>(in_bufinfo.ptr);
        for (py::ssize_t i = 0; i < steps; ++i) {
            // uint8 is promoted to uint32
            count += POPCOUNT_32(in_ptr[i]);
        }
    }
    return count;
}

// TODO: Currently this is pretty slow unless hitting the "uint64_t" branch,
// maybe two pass approach? first compute all popcounts, then sum (Numpy does
// this). Maybe the additions could be auto-vectorized in that case.
// TODO: Refactor this code to use a templated function with uint8_t / uint64_t
py::array_t<uint32_t> _popcount_2d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
        arr) {
    // Input buffer set-up
    py::buffer_info in_bufinfo = arr.request();
    if (in_bufinfo.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    const py::ssize_t n_samples = in_bufinfo.shape[0];

    // Input buffer set-up
    auto out = py::array_t<uint32_t>(n_samples);
    auto out_ptr = static_cast<uint32_t*>(out.request().ptr);
    std::fill(out_ptr, out_ptr + n_samples, 0);

#ifdef DEBUG_LOGS
    std::cout << "in_buf_ptr addr: "
              << reinterpret_cast<std::uintptr_t>(in_bufinfo.ptr) << std::endl;
    std::cout << "uint64_t alignment requirement: " << alignof(uint64_t)
              << std::endl;
    std::cout << "Allignment check (coorect if 0): "
              << reinterpret_cast<std::uintptr_t>(in_bufinfo.ptr) %
                     alignof(uint64_t)
              << std::endl;
#endif
    // Conversion between const void* and const std::uintptr_t requires reinterpret
    if (reinterpret_cast<std::uintptr_t>(in_bufinfo.ptr) % alignof(uint64_t) ==
        0) {
#ifdef DEBUG_LOGS
        std::cout << "DEBUG: _popcount_2d triggered uint64 + popcount 64 branch"
                  << std::endl;
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        const py::ssize_t steps = in_bufinfo.shape[1] / sizeof(uint64_t);
        const py::ssize_t stride = in_bufinfo.strides[0] / sizeof(uint64_t);
        auto in_ptr = static_cast<const uint64_t*>(in_bufinfo.ptr);
        for (py::ssize_t i = 0; i < n_samples; ++i) {
            const uint64_t* row_ptr = in_ptr + i * stride;
            for (py::ssize_t j = 0; j < steps; ++j) {
                out_ptr[i] += POPCOUNT_64(row_ptr[j]);
            }
        }
    } else {
#ifdef DEBUG_LOGS
        std::cout << "DEBUG: _popcount_2d triggered uint8 + popcount 32 branch"
                  << std::endl;
#endif
        // Misaligned, loop over bytes
        const py::ssize_t steps = in_bufinfo.shape[1];
        const py::ssize_t stride = in_bufinfo.strides[0];
        auto in_ptr = static_cast<const uint8_t*>(in_bufinfo.ptr);
        for (py::ssize_t i = 0; i < n_samples; ++i) {
            const uint8_t* row_ptr = in_ptr + i * stride;
            for (py::ssize_t j = 0; j < steps; ++j) {
                out_ptr[i] += POPCOUNT_32(row_ptr[j]);
            }
        }
    }
    return out;
}
// TODO: This works but it is *extremely slow* so for now it is completely
// avoided in the code and done in python instead
// TODO: This *should be done using a lookup table*, which would be significantly faster
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

// Contraint: T must be uint64_t or uint8_t
template <typename T>
void _calc_arr_vec_jt(const py::buffer_info& arr_buf,
                      const py::buffer_info& vec_buf,
                      const py::ssize_t n_samples, const uint32_t vec_popcount,
                      const py::array_t<uint32_t>& cardinalities,
                      py::array_t<double>& out) {
    const py::ssize_t steps = arr_buf.shape[1] / sizeof(T);
    const py::ssize_t stride = arr_buf.strides[0] / sizeof(T);
    auto arr_ptr = static_cast<const T*>(arr_buf.ptr);
    auto vec_ptr = static_cast<const T*>(vec_buf.ptr);
    auto out_ptr = static_cast<double*>(out.request().ptr);
    auto card_ptr = static_cast<const uint32_t*>(cardinalities.request().ptr);

    for (py::ssize_t i = 0; i < n_samples; ++i) {
        const T* arr_row_ptr = arr_ptr + i * stride;
        uint32_t intersection{0};
        for (py::ssize_t j = 0; j < steps; ++j) {
            if constexpr (std::is_same_v<T, uint64_t>) {
                intersection += POPCOUNT_64(arr_row_ptr[j] & vec_ptr[j]);
            } else {
                intersection += POPCOUNT_32(arr_row_ptr[j] & vec_ptr[j]);
            }
        }
        auto denominator = card_ptr[i] + vec_popcount - intersection;
        out_ptr[i] =
            intersection / std::max(static_cast<double>(denominator), 1.0);
    }
}

// # NOTE: This function is the bottleneck for bb compute calculations
// In this function, _popcount_2d takes around ~25% of the time, _popcount_1d
// around 5%. The internal loop with the popcounts is also quite heavy.
// TODO: Investigate simple SIMD vectorization of these loops
py::array_t<double> jt_sim_packed(
    py::array_t<uint8_t> arr, py::array_t<uint8_t> vec,
    std::optional<py::array_t<uint32_t>> cardinalities_opt) {
    py::buffer_info arr_bufinfo = arr.request();
    py::buffer_info vec_bufinfo = vec.request();

    if (arr_bufinfo.ndim != 2 || vec_bufinfo.ndim != 1) {
        throw std::runtime_error("arr must be 2D and vec must be 1D");
    }
    if (arr_bufinfo.shape[1] != vec_bufinfo.shape[0]) {
        throw std::runtime_error("arr and vec have incompatible shapes");
    }

    py::ssize_t n_samples = arr_bufinfo.shape[0];

    py::array_t<uint32_t> cardinalities;
    if (cardinalities_opt) {
        cardinalities = cardinalities_opt.value();
    } else {
        cardinalities = _popcount_2d(arr);
    }
    uint32_t vec_popcount = _popcount_1d(vec);

    bool arr_is_8byte_aligned =
        reinterpret_cast<std::uintptr_t>(arr_bufinfo.ptr) % alignof(uint64_t) ==
        0;
    bool vec_is_8byte_aligned =
        reinterpret_cast<std::uintptr_t>(vec_bufinfo.ptr) % alignof(uint64_t) ==
        0;

    auto out = py::array_t<double>(n_samples);
    if (arr_is_8byte_aligned && vec_is_8byte_aligned) {
#ifdef DEBUG_LOGS
        std::cout
            << "DEBUG: jt_sim_packed fn triggered uint64 + popcount 64 branch"
            << std::endl;
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        _calc_arr_vec_jt<uint64_t>(arr_bufinfo, vec_bufinfo, n_samples,
                                   vec_popcount, cardinalities, out);
    } else {
#ifdef DEBUG_LOGS
        std::cout
            << "DEBUG: jt_sim_packed fn triggered uint8 + popcount 32 branch"
            << std::endl;
#endif
        // Misaligned, loop over bytes
        _calc_arr_vec_jt<uint8_t>(arr_bufinfo, vec_bufinfo, n_samples,
                                  vec_popcount, cardinalities, out);
    }
    return out;
}

py::tuple jt_most_dissimilar_packed_also_requiring_unpacked(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> Y,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
        Y_unpacked) {
    py::buffer_info Y_buf = Y.request();
    if (Y_buf.ndim != 2) {
        throw std::runtime_error("Y must be a 2D array");
    }
    py::ssize_t n_samples = Y_buf.shape[0];
    py::ssize_t n_features_packed = Y_buf.shape[1];
    auto Y_ptr = static_cast<const uint8_t*>(Y_buf.ptr);

    py::buffer_info Y_unpacked_buf = Y_unpacked.request();
    py::ssize_t n_features = Y_unpacked_buf.shape[1];
    auto Y_unpacked_ptr = static_cast<const uint8_t*>(Y_unpacked_buf.ptr);

    auto linear_sum = py::array_t<uint64_t>(n_features);
    auto linear_sum_ptr = static_cast<uint64_t*>(linear_sum.request().ptr);
    std::fill(linear_sum_ptr, linear_sum_ptr + n_features, 0);

    // TODO: This sum could be vectorized or done more efficiently
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

    m.def("_popcount_2d", &_popcount_2d, "2D popcount", py::arg("a"));
    m.def("_popcount_1d", &_popcount_1d, "1D popcount", py::arg("a"));

    m.def("jt_isim", &jt_isim, "iSIM Tanimoto calculation", py::arg("c_total"),
          py::arg("n_objects"));

    m.def("jt_sim_packed", &jt_sim_packed,
          "Tanimoto similarity between a matrix of packed fps and a single "
          "packed fp",
          py::arg("arr"), py::arg("vec"),
          py::arg("_cardinalities") = std::nullopt);

    m.def("jt_most_dissimilar_packed_also_requiring_unpacked",
          &jt_most_dissimilar_packed_also_requiring_unpacked,
          "Finds two fps in a packed fp array that are the most "
          "Tanimoto-dissimilar",
          py::arg("Y"), py::arg("Y_unpacked"));
}
