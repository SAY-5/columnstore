#include "core/batch.h"

namespace columnstore {

std::size_t popcount_bitmap(const uint8_t* bm, std::size_t n) {
    const std::size_t full_bytes = n / 8;
    std::size_t count = 0;
    for (std::size_t i = 0; i < full_bytes; ++i) {
        count += static_cast<std::size_t>(__builtin_popcount(bm[i]));
    }
    const std::size_t leftover = n % 8;
    if (leftover != 0) {
        const uint8_t mask = static_cast<uint8_t>((1u << leftover) - 1u);
        count += static_cast<std::size_t>(__builtin_popcount(bm[full_bytes] & mask));
    }
    return count;
}

} // namespace columnstore
