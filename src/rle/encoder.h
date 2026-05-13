#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/column.h"

namespace columnstore {

// RLE encode a contiguous int32 buffer. Each adjacent run of equal values is
// collapsed into a (value, length) pair. Run lengths are clamped to
// UINT32_MAX; in practice no run exceeds the source buffer size.
struct RleEncodedInt32 {
    std::vector<int32_t> values;
    std::vector<uint32_t> lengths;
};

RleEncodedInt32 rle_encode_int32(const int32_t* data, std::size_t n);

// Convenience: encode and stuff the result into a Column<int32_t>.
Column<int32_t> rle_encode_column(const std::vector<int32_t>& data);

} // namespace columnstore
