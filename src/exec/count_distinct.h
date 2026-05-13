#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "core/dict.h"

namespace columnstore {

// CountDistinct returns the number of distinct values seen in a column.
//
// Fast path: when the column is dictionary-encoded with cardinality K <= 256,
// the answer is exactly K (or the count of dict entries actually referenced
// when an upstream filter is in play). Either way, the cost is O(K) not
// O(N). For an N=1M, K=8 column that is a ~125,000x reduction in work.
//
// Scalar fallback: walks the raw column into an std::unordered_set.
class CountDistinctInt32 {
public:
    // Fast path: dict-encoded, no filter -> answer is the dictionary size.
    static std::size_t run_dict(const DictColumn<int32_t>& d) { return d.cardinality(); }

    // Fast path with selection: count the number of unique dict codes
    // referenced under a row-level keep bitmap. Still O(K + N/8) because we
    // must scan the bitmap, but only K dict slots and a small bool array.
    static std::size_t
    run_dict_filtered(const DictColumn<int32_t>& d, const uint8_t* keep_bm, std::size_t n) {
        const std::size_t K = d.cardinality();
        std::vector<uint8_t> seen(K, 0);
        const uint8_t* codes = d.codes_data();
        for (std::size_t i = 0; i < n; ++i) {
            const bool keep = (keep_bm[i / 8] >> (i % 8)) & 1u;
            if (keep) {
                seen[codes[i]] = 1;
            }
        }
        std::size_t out = 0;
        for (uint8_t s : seen) {
            out += s;
        }
        return out;
    }

    // Scalar fallback: walk raw values into a hash set. O(N) expected.
    static std::size_t run_scalar(const int32_t* data, std::size_t n) {
        std::unordered_set<int32_t> s;
        s.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            s.insert(data[i]);
        }
        return s.size();
    }
};

// Dict-encoded equality filter (codes-buffer fast path).
//
// On AVX2 hosts this uses _mm256_cmpeq_epi8 to compare 32 codes per
// instruction against a splatted target byte, which is much faster than the
// int32 path (32 lanes vs 8 lanes per cycle). Returns the number of set
// bits written into `out_bm`.
std::size_t
filter_dict_eq_scalar(const uint8_t* codes, std::size_t n, uint8_t target, uint8_t* out_bm);

#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
std::size_t
filter_dict_eq_avx2(const uint8_t* codes, std::size_t n, uint8_t target, uint8_t* out_bm);
#endif

} // namespace columnstore
