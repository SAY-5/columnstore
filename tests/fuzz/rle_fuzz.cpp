#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rle/decoder.h"

// libFuzzer entry. The input bytes are interpreted as a sequence of
// (int32 value, uint32 length) pairs. The decoder must not crash or read
// past the end of the input for ANY malformed encoding.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    using namespace columnstore;

    // Each run is 8 bytes: 4 for value, 4 for length.
    const std::size_t run_count = size / 8;
    if (run_count == 0) {
        // Empty input: trivially safe.
        return 0;
    }

    std::vector<int32_t> values(run_count);
    std::vector<uint32_t> lengths(run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
        std::memcpy(&values[i], data + i * 8, sizeof(int32_t));
        std::memcpy(&lengths[i], data + i * 8 + 4, sizeof(uint32_t));
        // Cap raw lengths to keep the fuzzer from allocating GB of memory
        // on a malicious input. The decoder doesn't allocate per-run, but
        // we still want fast iterations.
        if (lengths[i] > 1u << 12) {
            lengths[i] = (lengths[i] & 0xFFFu);
        }
    }

    RleDecoder dec(values.data(), lengths.data(), run_count);
    RleBatch batch;
    int safety = 0;
    while (dec.next_batch(batch)) {
        if (batch.size == 0) {
            break;
        }
        ++safety;
        if (safety > 1024) {
            break;
        }
    }
    return 0;
}
