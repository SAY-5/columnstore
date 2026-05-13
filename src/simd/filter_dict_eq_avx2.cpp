#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "exec/count_distinct.h"

namespace columnstore {

// Compare 32 dictionary codes per AVX2 instruction against a target byte.
// Each compare lane is 0xFF when codes[i] == target, else 0x00.
// _mm256_movemask_epi8 collapses 32 lanes into a 32-bit mask, written
// directly as 4 packed bytes of bitmap (LSB-first matches our convention).
std::size_t
filter_dict_eq_avx2(const uint8_t* codes, std::size_t n, uint8_t target, uint8_t* out_bm) {
    const std::size_t bytes = (n + 7) / 8;
    for (std::size_t b = 0; b < bytes; ++b) {
        out_bm[b] = 0;
    }
    const __m256i splat = _mm256_set1_epi8(static_cast<int8_t>(target));

    std::size_t i = 0;
    std::size_t cnt = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + i));
        const __m256i eq = _mm256_cmpeq_epi8(v, splat);
        const uint32_t m = static_cast<uint32_t>(_mm256_movemask_epi8(eq));
        out_bm[(i / 8) + 0] = static_cast<uint8_t>(m & 0xFF);
        out_bm[(i / 8) + 1] = static_cast<uint8_t>((m >> 8) & 0xFF);
        out_bm[(i / 8) + 2] = static_cast<uint8_t>((m >> 16) & 0xFF);
        out_bm[(i / 8) + 3] = static_cast<uint8_t>((m >> 24) & 0xFF);
        cnt += static_cast<std::size_t>(__builtin_popcount(m));
    }
    for (; i < n; ++i) {
        if (codes[i] == target) {
            out_bm[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            ++cnt;
        }
    }
    return cnt;
}

} // namespace columnstore

#endif
