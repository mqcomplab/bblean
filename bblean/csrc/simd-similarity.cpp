#include <nmmintrin.h>  // SSE
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

namespace py = pybind11;

// Numpy is around 2x faster if not using SIMD
// nocona does *not* have sse3, required for _mm_shuffle_epi8, but most modern
// CPU do, so turn SSE3 *on*
//
// AVX2 implementation would be *significantly* faster (2x)
// TODO this will *not* compile on a mac if using AVX2
// TODO this will *not* compile on a mac if using SSE2
// Probably the best thing to do is to have the sse functions in a different
// fingerprints.cpp file, and raise when that module is imported on a mac, so
// the functions are not even attempted in that case (they would be super slow
// and I'm lazy to code for them)
//
// TODO: Use sse2neon to support Apple ARM chips (Mn)

// Double reverse mask
const __m128i DOUBLE_REVERSE_MASK =
    _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
void pack_bits_simd_sse(const uint8_t* ptr_in, uint8_t* ptr_out,
                        size_t n2_bytes) {
    for (size_t i = 0; i != n2_bytes; i += 16) {
        // load 16 0|1-bytes from ptr_in into an SSE register
        // (unaligned load, still fast)
        __m128i input = _mm_loadu_si128((__m128i*)(ptr_in + i));

        // Convert to LSB:
        // Flip the loaded bytes using the DOUBLE_REVERSE_MASK
        __m128i reversed_input = _mm_shuffle_epi8(input, DOUBLE_REVERSE_MASK);
        // Compare each byte to zero -> 0xFF if nonzero, 0x00 if zero
        __m128i mask = _mm_cmpgt_epi8(reversed_input, _mm_setzero_si128());

        // Pack the 16 bits into a single byte, extracting the MSB of each
        uint16_t bytes2 = (uint16_t)_mm_movemask_epi8(mask);

        // LSB magic invocation can be used here instead of _mm_shuffle_epi8,
        // but it is way slower ptr_out[i] = ((byte * 0x0202020202ULL &
        // 0x010884422010ULL) % 1023) & 0xFF;
        *(uint16_t*)(ptr_out + i / 8) = bytes2;
    }
}

py::array_t<uint8_t> _pack_fingerprints_1d(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
        arr) {
    py::buffer_info in_bufinfo = arr.request();
    if (in_bufinfo.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    const int n_features = in_bufinfo.shape[0];
    if (n_features % 8 != 0) {
        throw std::runtime_error("Only n_features divisible by 8 is supported");
    }
    const int n_bytes = n_features / 8;
    auto ptr_in = arr.data();
    py::array_t<uint8_t> out(n_bytes);  // Init to zeros
    auto ptr_out = out.mutable_data();
    pack_bits_simd_sse(ptr_in, ptr_out, n_bytes * 2);
    return out;
}

PYBIND11_MODULE(_cpp_simd, m) {
    m.doc() = "SIMD C++ extensions";

    m.def("_pack_fingerprints_1d", &_pack_fingerprints_1d, "2D popcount",
          py::arg("arr"));
}
