#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/types.h"

namespace columnstore {

// Decoded outcome of one RleDecoder::next_batch call.
struct RleBatch {
    int32_t buffer[kBatchSize];
    std::size_t size = 0; // 0 -> stream exhausted
};

// Stateful RLE decoder. Designed to feed the vectorized execution model:
// pull one batch (up to kBatchSize values) per call, decoding lazily so we
// never need to materialize the whole column into memory.
class RleDecoder {
public:
    RleDecoder(const int32_t* values, const uint32_t* lengths, std::size_t run_count);

    // Returns false if the RLE stream is structurally invalid (e.g. a zero
    // run length appears past index 0). On success, fills `out` with up to
    // kBatchSize values and sets out.size; size==0 indicates EOF.
    bool next_batch(RleBatch& out);

    // Whether the decoder has detected an inconsistency in its inputs. The
    // fuzzer harness uses this to assert "malformed input never crashes".
    bool malformed() const { return malformed_; }

private:
    const int32_t* values_;
    const uint32_t* lengths_;
    std::size_t run_count_;
    std::size_t run_idx_ = 0;
    uint32_t remaining_in_run_ = 0;
    bool malformed_ = false;
};

} // namespace columnstore
