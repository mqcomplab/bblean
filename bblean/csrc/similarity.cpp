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

// TODO: See if worth it to use vector popcount intrinsics (AVX-512, only some
// CPU)
// TODO: Refactor this code to use a templated function with uint8_t / uint64_t
// like jt_sim_packed
namespace py = pybind11;

auto print_8byte_alignment_check(const py::array_t<uint8_t>& arr) -> void {
    py::print("arr buf addr: ", reinterpret_cast<std::uintptr_t>(arr.data()));
    py::print("uint64_t alignment requirement: ", alignof(uint64_t));
    py::print("Alignment check (coorect if 0): ", reinterpret_cast<std::uintptr_t>(arr.data()) % alignof(uint64_t));
}

// TODO: we can should be able to assume the input array size is always a
// multiple of 64, otherwise the hand-coded C++ code-path should not be
// triggered (?)
uint32_t _popcount_1d(const py::array_t<uint8_t>& arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
#ifdef DEBUG_LOGS
    print_8byte_alignment_check(arr);
#endif
    uint32_t count{0};  // Output scalar
    // Convert between ptr and integer requires reinterpret
    bool is_8byte_aligned =
        reinterpret_cast<std::uintptr_t>(arr.data()) % alignof(uint64_t) == 0;
    py::ssize_t steps = arr.shape(0);
    if (is_8byte_aligned) {
#ifdef DEBUG_LOGS
        py::print("DEBUG: _popcount_1d fn triggered uint64 + popcount 64");
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        steps /= sizeof(uint64_t);
        auto ptr_in = static_cast<const uint64_t*>(arr.request().ptr);
        for (py::ssize_t i = 0; i != steps; ++i) {
            count += POPCOUNT_64(ptr_in[i]);
        }
    } else {
#ifdef DEBUG_LOGS
        py::print("DEBUG: _popcount_1d fn triggered uint8 + popcount 32");
#endif
        // Misaligned, loop over bytes
        auto ptr_in = arr.data();
        for (py::ssize_t i = 0; i != steps; ++i) {
            count += POPCOUNT_32(ptr_in[i]);  // uint8 promoted to uint32
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
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    const py::ssize_t n_samples = arr.shape(0);

    auto out = py::array_t<uint32_t>(n_samples);
    auto out_ptr = out.mutable_data();
    std::memset(out_ptr, 0, out.nbytes());

#ifdef DEBUG_LOGS
    print_8byte_alignment_check(arr);
#endif
    py::ssize_t stride = arr.strides(0);
    py::ssize_t steps = arr.shape(1);
    // Convert between ptr and integer requires reinterpret
    bool is_8byte_aligned =
        reinterpret_cast<std::uintptr_t>(arr.data()) % alignof(uint64_t) == 0;
    if (is_8byte_aligned) {
#ifdef DEBUG_LOGS
        py::print("DEBUG: _popcount_2d fn triggered uint64 + popcount 64");
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        stride /= sizeof(uint64_t);
        steps /= sizeof(uint64_t);
        auto ptr_in = static_cast<const uint64_t*>(arr.request().ptr);
        for (py::ssize_t i = 0; i != n_samples; ++i) {
            const uint64_t* row_ptr = ptr_in + i * stride;
            for (py::ssize_t j = 0; j != steps; ++j) {
                out_ptr[i] += POPCOUNT_64(row_ptr[j]);
            }
        }
    } else {
#ifdef DEBUG_LOGS
        py::print("DEBUG: _popcount_2d fn triggered uint8 + popcount 32");
#endif
        // Misaligned, loop over bytes
        auto in_ptr = arr.data();
        for (py::ssize_t i = 0; i != n_samples; ++i) {
            const uint8_t* row_ptr = in_ptr + i * stride;
            for (py::ssize_t j = 0; j != steps; ++j) {
                out_ptr[i] += POPCOUNT_32(row_ptr[j]);
            }
        }
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

// TODO: Remove code duplication
py::array_t<uint8_t> _nochecks_unpack_fingerprints_1d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
        packed_fps,
    std::optional<py::ssize_t> n_features_opt) {
    py::ssize_t n_bytes = packed_fps.shape(1);
    py::ssize_t n_features = n_features_opt.value_or(n_bytes * 8);
    if (n_features % 8 != 0) {
        throw std::runtime_error("Only n_features divisible by 8 is supported");
    }
    auto out = py::array_t<uint8_t>(n_features);
    auto out_ptr = out.mutable_data();
    auto in_ptr = packed_fps.data();
    for (py::ssize_t j = 0; j != n_features; j += 8) {
        // Copy the next 8 uint8 values in one go
        std::memcpy(out_ptr + j, BYTE_TO_BITS[in_ptr[j / 8]].data(), 8);
    }
    return out;
}

py::array_t<uint8_t> _nochecks_unpack_fingerprints_2d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
        packed_fps,
    std::optional<py::ssize_t> n_features_opt) {
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

    for (py::ssize_t i = 0; i != n_samples; ++i) {
        for (py::ssize_t j = 0; j != n_features; j += 8) {
            // Copy the next 8 uint8 values in one go
            std::memcpy(&acc_out(i, j), BYTE_TO_BITS[acc_in(i, j / 8)].data(),
                        8);
        }
    }
    return out;
}

// Wrapper over _nochecks_unpack_fingerprints that performs ndim checks
py::array_t<uint8_t> unpack_fingerprints(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
        packed_fps,
    std::optional<py::ssize_t> n_features_opt) {
    if (packed_fps.ndim() == 1) {
        return _nochecks_unpack_fingerprints_1d(packed_fps, n_features_opt);
    }
    if (packed_fps.ndim() == 2) {
        return _nochecks_unpack_fingerprints_2d(packed_fps, n_features_opt);
    }
    throw std::runtime_error("Input array must be 1- or 2-dimensional");
}

// TODO: Allow multiple dtypes as input?
py::array_t<uint8_t> calc_centroid(
    const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>&
        linear_sum,
    int64_t n_samples, bool pack = true) {
    if (linear_sum.ndim() != 1) {
        throw std::runtime_error("linear_sum must be 1-dimensional");
    }

    py::ssize_t n_features = linear_sum.shape(0);
    auto ptr_linear_sum = linear_sum.data();

    py::array_t<uint8_t> centroid_unpacked(n_features);
    auto ptr_centroid_unpacked = centroid_unpacked.mutable_data();
    if (n_samples <= 1) {
        for (int i{0}; i != n_features; ++i) {
            // Cast not required, but added for clarity since this is a
            // narrowing conversion. if n_samples <= 1 then linear_sum is
            // guaranteed to have a value that a uint8_t can hold (it should be
            // 0 or 1)
            ptr_centroid_unpacked[i] = static_cast<uint8_t>(ptr_linear_sum[i]);
        }
    } else {
        auto threshold = n_samples * 0.5;
        for (int i{0}; i != n_features; ++i) {
            ptr_centroid_unpacked[i] = (ptr_linear_sum[i] >= threshold) ? 1 : 0;
        }
    }

    if (not pack) {
        return centroid_unpacked;
    }

    auto const_ptr_centroid_unpacked = centroid_unpacked.data();
    int n_bytes = (n_features + 7) / 8;
    auto centroid_packed = py::array_t<uint8_t>(n_bytes);
    auto ptr_centroid_packed = centroid_packed.mutable_data();
    std::memset(ptr_centroid_packed, 0, centroid_packed.nbytes());

    // Slower than numpy, due to lack of SIMD
    for (int i{0}; i != n_features; ++i) {
        if (const_ptr_centroid_unpacked[i]) {
            ptr_centroid_packed[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
    return centroid_packed;
}

double jt_isim(
    const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>&
        linear_sum,
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

    auto in_ptr = linear_sum.data();
    uint64_t sum_kq{0};
    for (py::ssize_t i = 0; i != n_features; ++i) {
        sum_kq += in_ptr[i];
    }

    if (sum_kq == 0) {
        return 1.0;
    }

    uint64_t sum_kqsq{0};
    for (py::ssize_t i = 0; i != n_features; ++i) {
        sum_kqsq += in_ptr[i] * in_ptr[i];
    }
    auto a = (sum_kqsq - sum_kq) / 2.0;
    return a / ((a + (n_objects * sum_kq)) - sum_kqsq);
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
    auto out_ptr = out.mutable_data();
    auto card_ptr = cardinalities.data();

    for (py::ssize_t i = 0; i != n_samples; ++i) {
        const T* arr_row_ptr = arr_ptr + i * stride;
        uint32_t intersection{0};
        for (py::ssize_t j = 0; j != steps; ++j) {
            if constexpr (std::is_same_v<T, uint64_t>) {
                intersection += POPCOUNT_64(arr_row_ptr[j] & vec_ptr[j]);
            } else {
                intersection += POPCOUNT_32(arr_row_ptr[j] & vec_ptr[j]);
            }
        }
        auto denominator = card_ptr[i] + vec_popcount - intersection;
        // Cast is technically unnecessary since std::max promotes to double,
        // but added here for clarity
        out_ptr[i] =
            intersection / std::max(static_cast<double>(denominator), 1.0);
    }
}

// # NOTE: This function is the bottleneck for bb compute calculations
// In this function, _popcount_2d takes around ~25% of the time, _popcount_1d
// around 5%. The internal loop with the popcounts is also quite heavy.
// TODO: Investigate simple SIMD vectorization of these loops
// TODO cardinalities should not be a copy, also does this function return a
// copy?!
py::array_t<double> jt_sim_packed(
    const py::array_t<uint8_t>& arr, const py::array_t<uint8_t>& vec,
    std::optional<py::array_t<uint32_t>> cardinalities_opt) {
    if (arr.ndim() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("arr must be 2D, vec must be 1D");
    }
    if (arr.shape(1) != vec.shape(0)) {
        throw std::runtime_error(
            "Shapes should be (N, F) for arr and (F,) for vec");
    }

    py::array_t<uint32_t> cardinalities =
        cardinalities_opt.value_or(_popcount_2d(arr));
    uint32_t vec_popcount = _popcount_1d(vec);

    py::ssize_t n_samples = arr.shape(0);
    py::buffer_info arr_bufinfo = arr.request();
    py::buffer_info vec_bufinfo = vec.request();
    auto out = py::array_t<double>(n_samples);

    bool arr_is_8byte_aligned =
        reinterpret_cast<std::uintptr_t>(arr.data()) % alignof(uint64_t) == 0;
    bool vec_is_8byte_aligned =
        reinterpret_cast<std::uintptr_t>(vec.data()) % alignof(uint64_t) == 0;
    if (arr_is_8byte_aligned && vec_is_8byte_aligned) {
#ifdef DEBUG_LOGS
        py::print("DEBUG: jt_sim_packed fn triggered uint64 + popcount 64");
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        _calc_arr_vec_jt<uint64_t>(arr_bufinfo, vec_bufinfo, n_samples,
                                   vec_popcount, cardinalities, out);
    } else {
#ifdef DEBUG_LOGS
        py::print("DEBUG: jt_sim_packed fn triggered uint8 + popcount 32");
#endif
        // Misaligned, loop over bytes
        _calc_arr_vec_jt<uint8_t>(arr_bufinfo, vec_bufinfo, n_samples,
                                  vec_popcount, cardinalities, out);
    }
    return out;
}

// TODO: I believe strides are not necessary for contiguous arrays
py::tuple jt_most_dissimilar_packed(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> fps,
    std::optional<py::ssize_t> n_features_opt) {
    if (fps.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    py::ssize_t n_samples = fps.shape(0);
    py::ssize_t n_features_packed = fps.shape(1);
    py::ssize_t fps_strides0 = fps.strides(0);

    auto fps_unpacked = _nochecks_unpack_fingerprints_2d(fps, n_features_opt);
    py::ssize_t n_features = fps_unpacked.shape(1);
    py::ssize_t fps_unpacked_strides0 = fps_unpacked.strides(0);

    auto linear_sum = py::array_t<uint64_t>(n_features);
    auto linear_sum_ptr = linear_sum.mutable_data();
    std::memset(linear_sum_ptr, 0, linear_sum.nbytes());

    // TODO: This sum could be vectorized manually or automatically
    auto fps_unpacked_ptr = fps_unpacked.data();
    for (py::ssize_t i = 0; i != n_samples; ++i) {
        const uint8_t* row_ptr = fps_unpacked_ptr + i * fps_unpacked_strides0;
        for (py::ssize_t j = 0; j != n_features; ++j) {
            linear_sum_ptr[j] += row_ptr[j];
        }
    }

    auto packed_centroid = calc_centroid(linear_sum, n_samples, true);
    auto cardinalities = _popcount_2d(fps);

    auto sims_cent = jt_sim_packed(fps, packed_centroid, cardinalities);
    auto sims_cent_ptr = sims_cent.data();

    auto fps_ptr = fps.data();

    // argmin
    py::ssize_t fp1_idx = std::distance(
        sims_cent_ptr,
        std::min_element(sims_cent_ptr, sims_cent_ptr + n_samples));
    auto fp1_packed = py::array_t<uint8_t>(n_features_packed,
                                           fps_ptr + fp1_idx * fps_strides0);

    auto sims_fp1 = jt_sim_packed(fps, fp1_packed, cardinalities);
    auto sims_fp1_ptr = sims_fp1.data();

    // argmin
    py::ssize_t fp2_idx = std::distance(
        sims_fp1_ptr, std::min_element(sims_fp1_ptr, sims_fp1_ptr + n_samples));
    auto fp2_packed = py::array_t<uint8_t>(n_features_packed,
                                           fps_ptr + fp2_idx * fps_strides0);

    auto sims_fp2 = jt_sim_packed(fps, fp2_packed, cardinalities);

    return py::make_tuple(fp1_idx, fp2_idx, sims_fp1, sims_fp2);
}

PYBIND11_MODULE(_cpp_similarity, m) {
    m.doc() = "Optimized molecular similarity calculators (C++ extensions)";

    // Only bound for debugging purposes
    m.def("_nochecks_unpack_fingerprints_2d", &_nochecks_unpack_fingerprints_2d,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);
    m.def("_nochecks_unpack_fingerprints_1d", &_nochecks_unpack_fingerprints_1d,
          "Unpack packed fingerprints", py::arg("a"),
          py::arg("n_features") = std::nullopt);
    m.def("calc_centroid", &calc_centroid, "centroid calculation",
          py::arg("linear_sum"), py::arg("n_samples"), py::arg("pack") = true);
    m.def("_popcount_2d", &_popcount_2d, "2D popcount", py::arg("a"));
    m.def("_popcount_1d", &_popcount_1d, "1D popcount", py::arg("a"));

    // API
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
