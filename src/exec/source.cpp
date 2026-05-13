#include "exec/source.h"

namespace columnstore {

Int32ColumnScan::Int32ColumnScan(const int32_t* data, std::size_t n)
    : raw_data_(data), raw_size_(n) {}

Int32ColumnScan::Int32ColumnScan(const int32_t* values,
                                 const uint32_t* lengths,
                                 std::size_t run_count)
    : rle_mode_(true), decoder_(std::make_unique<RleDecoder>(values, lengths, run_count)) {}

std::optional<Batch<int32_t>> Int32ColumnScan::next() {
    if (rle_mode_) {
        if (!decoder_->next_batch(scratch_)) {
            return std::nullopt;
        }
        if (scratch_.size == 0) {
            return std::nullopt;
        }
        Batch<int32_t> b;
        b.data = scratch_.buffer;
        b.size = scratch_.size;
        return b;
    }
    if (cursor_ >= raw_size_) {
        return std::nullopt;
    }
    const std::size_t remaining = raw_size_ - cursor_;
    const std::size_t n = remaining < kBatchSize ? remaining : kBatchSize;
    Batch<int32_t> b;
    b.data = raw_data_ + cursor_;
    b.size = n;
    cursor_ += n;
    return b;
}

} // namespace columnstore
