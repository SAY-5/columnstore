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

// Sign-extend an __m256i of 8 int32 lanes to two __m256i of 4 int64 lanes,
// then add into a 4-lane int64 accumulator.
static inline __m256i add_extended(__m256i acc64, __m256i v) {
    const __m128i lo128 = _mm256_castsi256_si128(v);
    const __m128i hi128 = _mm256_extracti128_si256(v, 1);
    const __m256i lo64 = _mm256_cvtepi32_epi64(lo128);
    const __m256i hi64 = _mm256_cvtepi32_epi64(hi128);
    return _mm256_add_epi64(acc64, _mm256_add_epi64(lo64, hi64));
}

static inline int64_t hsum_epi64(__m256i v) {
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

int64_t sum_int32_avx2(const int32_t* data, std::size_t n, const uint8_t* bitmap) {
    int64_t acc = 0;
    std::size_t i = 0;
    (void)&hsum_epi32; // keep helper available for masked debugging path

    if (bitmap == nullptr) {
        // Unmasked path. Sign-extend each 8-lane vector to two 4-lane int64
        // vectors and accumulate directly in int64 to avoid per-lane epi32
        // overflow on adversarial bit patterns (e.g. 0xAAAAAAAA).
        __m256i acc64 = _mm256_setzero_si256();
        while (i + 8 <= n) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            acc64 = add_extended(acc64, v);
            i += 8;
        }
        acc += hsum_epi64(acc64);
        // Scalar tail.
        for (; i < n; ++i) {
            acc += static_cast<int64_t>(data[i]);
        }
        return acc;
    }

    // Masked path. For each 8-element chunk, splat the bitmap byte and
    // produce a per-lane keep mask. Use _mm256_and_si256 to zero out lanes
    // that don't pass. Accumulate in int64 to avoid per-lane overflow.
    __m256i acc64 = _mm256_setzero_si256();

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
        acc64 = add_extended(acc64, masked);
        i += 8;
    }
    acc += hsum_epi64(acc64);
    for (; i < n; ++i) {
        const bool keep = (bitmap[i / 8] >> (i % 8)) & 1u;
        if (keep) {
            acc += static_cast<int64_t>(data[i]);
        }
    }
    return acc;
}

} // namespace columnstore
