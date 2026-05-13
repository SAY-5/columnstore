#pragma once

#include <cstdint>
#include <vector>

#include "core/types.h"

namespace columnstore {

// A Batch is a contiguous slice of up to kBatchSize values from one column.
// Filter operators populate `selection` (a packed bitmap with one bit per row,
// LSB-first); downstream operators may use either the selection or a compacted
// indices array (consumers convert lazily).
template <typename T> struct Batch {
    const T* data = nullptr;    // points into the source column (zero-copy)
    std::size_t size = 0;       // number of valid values in this batch
    bool has_selection = false; // when true, selection bitmap is valid
    // Packed bitmap of size = (size + 7) / 8 bytes; bit k is the predicate
    // result for row k of this batch.
    std::vector<uint8_t> selection;

    static std::size_t bitmap_bytes(std::size_t n) { return (n + 7) / 8; }
};

// Helper: count set bits in a packed bitmap covering `n` bits.
std::size_t popcount_bitmap(const uint8_t* bm, std::size_t n);

} // namespace columnstore
