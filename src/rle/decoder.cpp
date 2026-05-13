#include "rle/decoder.h"

namespace columnstore {

RleDecoder::RleDecoder(const int32_t* values, const uint32_t* lengths, std::size_t run_count)
    : values_(values), lengths_(lengths), run_count_(run_count) {
    if (run_count_ > 0) {
        remaining_in_run_ = lengths_[0];
        if (remaining_in_run_ == 0) {
            malformed_ = true;
        }
    }
}

bool RleDecoder::next_batch(RleBatch& out) {
    out.size = 0;
    if (malformed_) {
        return false;
    }
    while (out.size < kBatchSize) {
        if (run_idx_ >= run_count_) {
            // EOF
            break;
        }
        if (remaining_in_run_ == 0) {
            ++run_idx_;
            if (run_idx_ >= run_count_) {
                break;
            }
            remaining_in_run_ = lengths_[run_idx_];
            if (remaining_in_run_ == 0) {
                malformed_ = true;
                return false;
            }
        }
        const int32_t v = values_[run_idx_];
        const std::size_t slack = kBatchSize - out.size;
        const std::size_t take =
            (remaining_in_run_ < slack) ? static_cast<std::size_t>(remaining_in_run_) : slack;
        for (std::size_t k = 0; k < take; ++k) {
            out.buffer[out.size + k] = v;
        }
        out.size += take;
        remaining_in_run_ -= static_cast<uint32_t>(take);
    }
    return true;
}

} // namespace columnstore
