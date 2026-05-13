#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "exec/operator.h"
#include "rle/decoder.h"

namespace columnstore {

// Scans an in-memory int32 column. Supports both raw (zero-copy slicing)
// and RLE-encoded inputs (decode-on-the-fly).
class Int32ColumnScan : public Int32Operator {
public:
    // Raw constructor. The pointer must outlive the scan.
    Int32ColumnScan(const int32_t* data, std::size_t n);

    // RLE constructor. The values/lengths pointers must outlive the scan.
    Int32ColumnScan(const int32_t* values, const uint32_t* lengths, std::size_t run_count);

    std::optional<Batch<int32_t>> next() override;

private:
    // Raw mode
    const int32_t* raw_data_ = nullptr;
    std::size_t raw_size_ = 0;
    std::size_t cursor_ = 0;

    // RLE mode
    bool rle_mode_ = false;
    std::unique_ptr<RleDecoder> decoder_;
    RleBatch scratch_;
};

} // namespace columnstore
