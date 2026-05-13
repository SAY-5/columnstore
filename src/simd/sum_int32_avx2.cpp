#include <immintrin.h>

#include <cstdint>

#include "simd/kernels.h"

namespace columnstore {

namespace {

// Horizontal reduction of a __m256i of 8 int32 lanes into one int64.
int64_t hsum_epi32(__m256i v) {
    // Extract low/high 128-bit halves, sign-extend each lane to int64, add.
    const __m128i lo128 = _mm256_castsi256_si128(v);
    const __m128i hi128 = _mm256_extracti128_si256(v, 1);

    const __m256i lo64 = _mm256_cvtepi32_epi64(lo128);
    const __m256i hi64 = _mm256_cvtepi32_epi64(hi128);
    const __m256i sum64 = _mm256_add_epi64(lo64, hi64);

    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), sum64);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

} // namespace

int64_t sum_int32_avx2(const int32_t* data, std::size_t n, const uint8_t* bitmap) {
    int64_t acc = 0;
    std::size_t i = 0;

    if (bitmap == nullptr) {
        // Unmasked path. Accumulate in epi32 across 8 lanes; flush to a
        // 64-bit accumulator every 256 elements to avoid overflow of any
        // single lane (which would need > 2^31 / 2^31 ~ infeasible, but the
        // periodic flush is cheap and removes the worry).
        __m256i lane_acc = _mm256_setzero_si256();
        std::size_t since_flush = 0;
        while (i + 8 <= n) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            lane_acc = _mm256_add_epi32(lane_acc, v);
            i += 8;
            since_flush += 8;
            if (since_flush >= 256) {
                acc += hsum_epi32(lane_acc);
                lane_acc = _mm256_setzero_si256();
                since_flush = 0;
            }
        }
        acc += hsum_epi32(lane_acc);
        // Scalar tail.
        for (; i < n; ++i) {
            acc += static_cast<int64_t>(data[i]);
        }
        return acc;
    }

    // Masked path. For each 8-element chunk, splat the bitmap byte and
    // produce a per-lane keep mask. Use _mm256_and_si256 to zero out lanes
    // that don't pass.
    __m256i lane_acc = _mm256_setzero_si256();
    std::size_t since_flush = 0;

    // Pre-computed per-lane bit masks: lane k keeps if bit k of the bitmap
    // byte is set.
    const __m256i lane_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    while (i + 8 <= n) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        const uint8_t byte = bitmap[i / 8];
        const __m256i splat = _mm256_set1_epi32(byte);
        const __m256i keep_bits = _mm256_and_si256(splat, lane_bits);
        const __m256i keep_mask = _mm256_cmpeq_epi32(keep_bits, lane_bits);
        const __m256i masked = _mm256_and_si256(v, keep_mask);
        lane_acc = _mm256_add_epi32(lane_acc, masked);
        i += 8;
        since_flush += 8;
        if (since_flush >= 256) {
            acc += hsum_epi32(lane_acc);
            lane_acc = _mm256_setzero_si256();
            since_flush = 0;
        }
    }
    acc += hsum_epi32(lane_acc);
    for (; i < n; ++i) {
        const bool keep = (bitmap[i / 8] >> (i % 8)) & 1u;
        if (keep) {
            acc += static_cast<int64_t>(data[i]);
        }
    }
    return acc;
}

} // namespace columnstore
