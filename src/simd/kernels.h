#pragma once

#include <cstddef>
#include <cstdint>

namespace columnstore {

// ---- Filter kernels ------------------------------------------------------
//
// `out_bitmap` must have at least (n + 7) / 8 bytes. Bit k (LSB-first within
// the byte) is set when `data[k] < threshold`. Returns the number of set
// bits.

std::size_t
filter_int32_lt_scalar(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap);

std::size_t
filter_int32_gt_scalar(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap);

#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
std::size_t
filter_int32_lt_avx2(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap);

std::size_t
filter_int32_gt_avx2(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap);
#endif

// ---- Sum kernels ---------------------------------------------------------
//
// Returns the sum of `data[k]` for which bit k of `bitmap` is set. If
// `bitmap` is null, all rows are summed. Accumulation happens in int64 to
// avoid overflow for typical workloads.

int64_t sum_int32_scalar(const int32_t* data, std::size_t n, const uint8_t* bitmap);

#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
int64_t sum_int32_avx2(const int32_t* data, std::size_t n, const uint8_t* bitmap);
#endif

// ---- Min/max/count kernels ----------------------------------------------

int32_t min_int32_scalar(const int32_t* data, std::size_t n, const uint8_t* bitmap, bool* empty);
int32_t max_int32_scalar(const int32_t* data, std::size_t n, const uint8_t* bitmap, bool* empty);
std::size_t count_int32_scalar(std::size_t n, const uint8_t* bitmap);

} // namespace columnstore
