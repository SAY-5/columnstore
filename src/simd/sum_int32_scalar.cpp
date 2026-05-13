#include "simd/kernels.h"

namespace columnstore {

int64_t sum_int32_scalar(const int32_t* data, std::size_t n, const uint8_t* bitmap) {
    int64_t acc = 0;
    if (bitmap == nullptr) {
        for (std::size_t i = 0; i < n; ++i) {
            acc += static_cast<int64_t>(data[i]);
        }
        return acc;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const bool keep = (bitmap[i / 8] >> (i % 8)) & 1u;
        if (keep) {
            acc += static_cast<int64_t>(data[i]);
        }
    }
    return acc;
}

int32_t min_int32_scalar(const int32_t* data, std::size_t n, const uint8_t* bitmap, bool* empty) {
    bool any = false;
    int32_t best = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool keep = (bitmap == nullptr) ? true : ((bitmap[i / 8] >> (i % 8)) & 1u) != 0;
        if (!keep) {
            continue;
        }
        if (!any || data[i] < best) {
            best = data[i];
            any = true;
        }
    }
    if (empty != nullptr) {
        *empty = !any;
    }
    return best;
}

int32_t max_int32_scalar(const int32_t* data, std::size_t n, const uint8_t* bitmap, bool* empty) {
    bool any = false;
    int32_t best = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool keep = (bitmap == nullptr) ? true : ((bitmap[i / 8] >> (i % 8)) & 1u) != 0;
        if (!keep) {
            continue;
        }
        if (!any || data[i] > best) {
            best = data[i];
            any = true;
        }
    }
    if (empty != nullptr) {
        *empty = !any;
    }
    return best;
}

std::size_t count_int32_scalar(std::size_t n, const uint8_t* bitmap) {
    if (bitmap == nullptr) {
        return n;
    }
    std::size_t c = 0;
    const std::size_t full = n / 8;
    for (std::size_t i = 0; i < full; ++i) {
        c += static_cast<std::size_t>(__builtin_popcount(bitmap[i]));
    }
    const std::size_t leftover = n % 8;
    if (leftover != 0) {
        const uint8_t mask = static_cast<uint8_t>((1u << leftover) - 1u);
        c += static_cast<std::size_t>(__builtin_popcount(bitmap[full] & mask));
    }
    return c;
}

} // namespace columnstore
