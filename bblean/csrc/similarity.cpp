#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "kernels.hpp"

// TODO: See if worth it to use vector popcount intrinsics (AVX-512, only some
// CPU) like jt_sim_packed
namespace py = pybind11;

template <typename T>
using CArrayForcecast =
    py::array_t<T, py::array::c_style | py::array::forcecast>;

// Packed fingerprints (1D or 2D) as rows of 64 bit words (see kernels.hpp).
// If the rows are whole, aligned words they are viewed in place. Otherwise
// they are copied, zero padded: all of them by words(), or a block of rows at a
// time by for_each_block(), so that large arrays are never copied whole.
class WordRows {
   public:
    explicit WordRows(const CArrayForcecast<uint8_t>& arr) {
        if (arr.ndim() != 1 && arr.ndim() != 2) {
            throw std::runtime_error("Input array must be 1- or 2-dimensional");
        }
        n = arr.ndim() == 2 ? static_cast<size_t>(arr.shape(0)) : 1;
        n_bytes_ = static_cast<size_t>(arr.shape(arr.ndim() - 1));
        w64 = (n_bytes_ + 7) / 8;
        src_ = arr.data();
        if (n_bytes_ % 8 == 0 &&
            reinterpret_cast<std::uintptr_t>(src_) % alignof(uint64_t) == 0) {
            view_ = reinterpret_cast<const uint64_t*>(src_);
        }
    }

    // All the rows (copied if they can't be viewed)
    const uint64_t* words() {
        if (view_ == nullptr) {
            copy_.resize(n * w64);
            copy_rows(0, n, copy_.data());
            view_ = copy_.data();
        }
        return view_;
    }

    // Calls f(rows, first, count) for consecutive blocks of count rows,
    // starting at row first
    template <typename F>
    void for_each_block(F&& f) const {
        if (view_ != nullptr) {
            f(view_, size_t{0}, n);
            return;
        }
        constexpr size_t kBlockRows = 1024;
        std::vector<uint64_t> block(std::min(n, kBlockRows) * w64);
        for (size_t first{0}; first < n; first += kBlockRows) {
            const size_t count = std::min(kBlockRows, n - first);
            copy_rows(first, count, block.data());
            f(block.data(), first, count);
        }
    }

    // Popcounts of count rows of these fingerprints, as words
    std::vector<uint32_t> popcounts(const uint64_t* rows, size_t count) const {
        std::vector<uint32_t> out(count);
        for (size_t i{0}; i != count; ++i) {
            out[i] = bblean::popcount(rows + i * w64, w64);
        }
        return out;
    }

    size_t n = 0;    // number of rows
    size_t w64 = 0;  // words per row

   private:
    void copy_rows(size_t first, size_t count, uint64_t* out) const {
        if (w64 == 0) {
            return;
        }
        for (size_t i{0}; i != count; ++i) {
            out[(i + 1) * w64 - 1] = 0;  // zero padding of the last word
            std::memcpy(out + i * w64, src_ + (first + i) * n_bytes_, n_bytes_);
        }
    }

    const uint8_t* src_ = nullptr;
    size_t n_bytes_ = 0;
    const uint64_t* view_ = nullptr;
    std::vector<uint64_t> copy_;
};

py::array_t<uint32_t> _popcount_2d(const CArrayForcecast<uint8_t>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    const WordRows rows(arr);
    py::array_t<uint32_t> out(rows.n);
    auto out_ptr = out.mutable_data();
    rows.for_each_block([&](const uint64_t* block, size_t first, size_t count) {
        for (size_t i{0}; i != count; ++i) {
            out_ptr[first + i] =
                bblean::popcount(block + i * rows.w64, rows.w64);
        }
    });
    return out;
}

uint32_t _popcount_1d(const CArrayForcecast<uint8_t>& arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    WordRows row(arr);
    return bblean::popcount(row.words(), row.w64);
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
    for (int i{0}; i != n_features; ++i) {
        centroid_unpacked_ptr[i] =
            bblean::centroid_value(linear_sum_cptr[i], n_samples);
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

    uint64_t sum_kqsq{0};
    for (py::ssize_t i{0}; i != n_features; ++i) {  // yes auto-vec by GCC
        sum_kqsq += in_cptr[i] * in_cptr[i];
    }
    return bblean::isim(sum_kq, sum_kqsq, n_objects);
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

py::array_t<double> _jt_sim_arr_vec_packed(
    const CArrayForcecast<uint8_t>& arr, const CArrayForcecast<uint8_t>& vec) {
    if (arr.ndim() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("arr must be 2D, vec must be 1D");
    }
    if (arr.shape(1) != vec.shape(0)) {
        throw std::runtime_error(
            "Shapes should be (N, F) for arr and (F,) for vec");
    }
    const WordRows rows(arr);
    WordRows query(vec);
    const uint64_t* query_words = query.words();
    py::array_t<double> out(rows.n);
    auto out_ptr = out.mutable_data();
    rows.for_each_block([&](const uint64_t* block, size_t first, size_t count) {
        const auto pops = rows.popcounts(block, count);
        bblean::jt_sims(block, pops.data(), count, rows.w64, query_words,
                        out_ptr + first);
    });
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
    const py::ssize_t n_features =
        n_features_opt.value_or(fps_packed.shape(1) * 8);
    if (n_features % 8 != 0) {
        throw std::runtime_error("Only features divisible by 8 is supported");
    }
    // Copied whole if the rows cannot be viewed as words
    WordRows rows(fps_packed);
    const uint64_t* words = rows.words();
    const auto pops = rows.popcounts(words, rows.n);
    py::array_t<double> sims1(rows.n), sims2(rows.n);
    // The centroid only uses the first n_features features
    const auto pair = bblean::most_dissimilar(
        words, pops.data(), rows.n, rows.w64, static_cast<size_t>(n_features),
        sims1.mutable_data(), sims2.mutable_data());
    return py::make_tuple(static_cast<py::ssize_t>(pair.fp1),
                          static_cast<py::ssize_t>(pair.fp2), sims1, sims2);
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
