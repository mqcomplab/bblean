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
// CPU) like jt_sim_packed
namespace py = pybind11;

template <typename T>
using CArrayForcecast =
    py::array_t<T, py::array::c_style | py::array::forcecast>;

auto is_8byte_aligned(const py::array_t<uint8_t>& a) -> bool {
    // Convert between ptr and integer requires reinterpret
    return reinterpret_cast<std::uintptr_t>(a.data()) % alignof(uint64_t) == 0;
}

auto print_8byte_alignment_check(const py::array_t<uint8_t>& arr) -> void {
    py::print("arr buf addr: ", reinterpret_cast<std::uintptr_t>(arr.data()));
    py::print("uint64_t alignment requirement: ", alignof(uint64_t));
    py::print("Is 8-byte aligned: ", is_8byte_aligned(arr));
}

uint32_t _popcount_1d(const py::array_t<uint8_t>& arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
#ifdef DEBUG_LOGS
    print_8byte_alignment_check(arr);
#endif
    uint32_t count{0};  // Output scalar
    py::ssize_t steps = arr.shape(0);
    if (is_8byte_aligned(arr) && (steps % 64 == 0)) {
#ifdef DEBUG_LOGS
        py::print("DEBUG: _popcount_1d fn triggered uint64 + popcount 64");
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        steps /= sizeof(uint64_t);
        auto in_cptr = static_cast<const uint64_t*>(arr.request().ptr);
        for (py::ssize_t i{0}; i != steps; ++i) {  // not auto-vec by GCC
            count += POPCOUNT_64(in_cptr[i]);
        }
        return count;
    }

#ifdef DEBUG_LOGS
    py::print("DEBUG: _popcount_1d fn triggered uint8 + popcount 32");
#endif
    // Misaligned, loop over bytes
    auto in_cptr = arr.data();
    for (py::ssize_t i{0}; i != steps; ++i) {  // not auto-vec by GCC
        count += POPCOUNT_32(in_cptr[i]);      // uint8 promoted to uint32
    }
    return count;
}

// TODO: Currently this is pretty slow unless hitting the "uint64_t" branch,
// maybe two pass approach? first compute all popcounts, then sum (Numpy does
// this). Maybe the additions could be auto-vec?
py::array_t<uint32_t> _popcount_2d(const CArrayForcecast<uint8_t>& arr) {
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
    py::ssize_t steps = arr.shape(1);
    if (is_8byte_aligned(arr) && (steps % 64 == 0)) {
#ifdef DEBUG_LOGS
        py::print("DEBUG: _popcount_2d fn triggered uint64 + popcount 64");
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        steps /= sizeof(uint64_t);
        auto in_cptr = static_cast<const uint64_t*>(arr.request().ptr);
        for (py::ssize_t i{0}; i != n_samples; ++i) {  // not auto-vec by GCC
            const uint64_t* row_cptr = in_cptr + i * steps;
            for (py::ssize_t j{0}; j != steps; ++j) {  // not auto-vec by GCC
                out_ptr[i] += POPCOUNT_64(row_cptr[j]);
            }
        }
        return out;
    }

#ifdef DEBUG_LOGS
    py::print("DEBUG: _popcount_2d fn triggered uint8 + popcount 32");
#endif
    // Misaligned, loop over bytes
    auto in_cptr = arr.data();
    for (py::ssize_t i{0}; i != n_samples; ++i) {  // not auto-vec by GCC
        const uint8_t* row_cptr = in_cptr + i * steps;
        for (py::ssize_t j{0}; j != steps; ++j) {  // not auto-vec by GCC
            out_ptr[i] += POPCOUNT_32(row_cptr[j]);
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

// Contraint: T must be uint64_t or uint8_t
template <typename T>
void _calc_arr_vec_jt(const py::array_t<uint8_t>& arr,
                      const py::array_t<uint8_t>& vec,
                      const py::ssize_t n_samples, const py::ssize_t n_features,
                      const uint32_t vec_popcount,
                      const py::array_t<uint32_t>& cardinalities,
                      py::array_t<double>& out) {
    const py::ssize_t steps = n_features / sizeof(T);
    auto arr_cptr = static_cast<const T*>(arr.request().ptr);
    auto vec_cptr = static_cast<const T*>(vec.request().ptr);
    auto card_cptr = cardinalities.data();
    auto out_ptr = out.mutable_data();

    for (py::ssize_t i{0}; i != n_samples; ++i) {  // not auto-vec by GCC
        const T* arr_row_cptr = arr_cptr + i * steps;
        uint32_t intersection{0};
        for (py::ssize_t j{0}; j != steps; ++j) {  // not auto-vec by GCC
            if constexpr (std::is_same_v<T, uint64_t>) {
                intersection += POPCOUNT_64(arr_row_cptr[j] & vec_cptr[j]);
            } else {
                intersection += POPCOUNT_32(arr_row_cptr[j] & vec_cptr[j]);
            }
        }
        auto denominator = card_cptr[i] + vec_popcount - intersection;
        // Cast is technically unnecessary since std::max promotes to double,
        // but added here for clarity (should compile to nop)
        out_ptr[i] =
            intersection / std::max(static_cast<double>(denominator), 1.0);
    }
}

// # NOTE: This function is the bottleneck for bb compute calculations
// In this function, _popcount_2d takes around ~25% of the time, _popcount_1d
// around 5%. The internal loop with the popcounts is also quite heavy.
// TODO: Investigate simple SIMD vectorization of these loops
// TODO: Does this function return a copy?
py::array_t<double> jt_sim_packed_precalc_cardinalities(
    const py::array_t<uint8_t>& arr, const py::array_t<uint8_t>& vec,
    const py::array_t<uint32_t>& cardinalities) {
    py::ssize_t n_samples = arr.shape(0);
    py::ssize_t n_features = arr.shape(1);
    if (arr.ndim() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("arr must be 2D, vec must be 1D");
    }
    if (n_features != vec.shape(0)) {
        throw std::runtime_error(
            "Shapes should be (N, F) for arr and (F,) for vec");
    }
    auto out = py::array_t<double>(n_samples);

    if (is_8byte_aligned(arr) && is_8byte_aligned(vec) &&
        (n_features % 64 == 0)) {
#ifdef DEBUG_LOGS
        py::print("DEBUG: jt_sim_packed fn triggered uint64 + popcount 64");
#endif
        // Aligned to 64-bit boundary, interpret as uint64_t
        _calc_arr_vec_jt<uint64_t>(arr, vec, n_samples, n_features,
                                   _popcount_1d(vec), cardinalities, out);
        return out;
    }

#ifdef DEBUG_LOGS
    py::print("DEBUG: jt_sim_packed fn triggered uint8 + popcount 32");
#endif
    // Misaligned, loop over bytes
    _calc_arr_vec_jt<uint8_t>(arr, vec, n_samples, n_features,
                              _popcount_1d(vec), cardinalities, out);
    return out;
}

py::array_t<double> _jt_sim_arr_vec_packed(const py::array_t<uint8_t>& arr,
                                           const py::array_t<uint8_t>& vec) {
    return jt_sim_packed_precalc_cardinalities(arr, vec, _popcount_2d(arr));
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
    auto fp1_packed = py::array_t<uint8_t>(
        n_features_packed, fps_packed_cptr + fp1_idx * n_features_packed);

    auto sims_fp1 = jt_sim_packed_precalc_cardinalities(fps_packed, fp1_packed,
                                                        cardinalities);
    auto sims_fp1_ptr = sims_fp1.data();

    // argmin
    py::ssize_t fp2_idx = std::distance(
        sims_fp1_ptr, std::min_element(sims_fp1_ptr, sims_fp1_ptr + n_samples));
    auto fp2_packed = py::array_t<uint8_t>(
        n_features_packed, fps_packed_cptr + fp2_idx * n_features_packed);

    auto sims_fp2 = jt_sim_packed_precalc_cardinalities(fps_packed, fp2_packed,
                                                        cardinalities);

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
