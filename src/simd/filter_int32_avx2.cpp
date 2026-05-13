#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "simd/kernels.h"

namespace columnstore {

namespace {

// Hot loop for `data[i] < threshold` (or `>` when LessThan is false).
// Processes 8 int32 per AVX2 vector. For each 8-lane chunk we extract a
// packed 8-bit mask via `_mm256_movemask_ps` after reinterpreting the
// compare result as float lanes. Eight chunks pack into one 64-bit word,
// written directly into the output bitmap.
template <bool LessThan>
std::size_t
filter_int32_avx2_impl(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap) {
    std::memset(out_bitmap, 0, (n + 7) / 8);

    const __m256i thr = _mm256_set1_epi32(threshold);
    std::size_t i = 0;

    // Main loop: 64 elements per iteration -> 8 bytes of bitmap.
    while (i + 64 <= n) {
        uint64_t bits = 0;
        for (int j = 0; j < 8; ++j) {
            const __m256i v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + j * 8));
            // _mm256_cmpgt_epi32(a, b) -> a > b (signed). For LessThan we
            // want data < threshold, equivalent to threshold > data.
            __m256i cmp;
            if (LessThan) {
                cmp = _mm256_cmpgt_epi32(thr, v);
            } else {
                cmp = _mm256_cmpgt_epi32(v, thr);
            }
            const int mask8 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            bits |= (static_cast<uint64_t>(mask8) & 0xFFull) << (j * 8);
        }
        std::memcpy(out_bitmap + i / 8, &bits, sizeof(bits));
        i += 64;
    }

    // Tail in chunks of 8 elements.
    while (i + 8 <= n) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i cmp;
        if (LessThan) {
            cmp = _mm256_cmpgt_epi32(thr, v);
        } else {
            cmp = _mm256_cmpgt_epi32(v, thr);
        }
        const int mask8 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        out_bitmap[i / 8] = static_cast<uint8_t>(mask8 & 0xFF);
        i += 8;
    }

    // Final scalar tail (< 8 elements).
    for (; i < n; ++i) {
        const bool pass = LessThan ? (data[i] < threshold) : (data[i] > threshold);
        if (pass) {
            out_bitmap[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
    }

    // Count set bits.
    std::size_t set_count = 0;
    const std::size_t full_bytes = n / 8;
    for (std::size_t b = 0; b < full_bytes; ++b) {
        set_count += static_cast<std::size_t>(__builtin_popcount(out_bitmap[b]));
    }
    const std::size_t leftover = n % 8;
    if (leftover != 0) {
        const uint8_t mask = static_cast<uint8_t>((1u << leftover) - 1u);
        set_count += static_cast<std::size_t>(__builtin_popcount(out_bitmap[full_bytes] & mask));
    }
    return set_count;
}

} // namespace

std::size_t
filter_int32_lt_avx2(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap) {
    return filter_int32_avx2_impl<true>(data, n, threshold, out_bitmap);
}

std::size_t
filter_int32_gt_avx2(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap) {
    return filter_int32_avx2_impl<false>(data, n, threshold, out_bitmap);
}

} // namespace columnstore
