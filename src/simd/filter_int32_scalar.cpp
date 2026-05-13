#include <cstring>

#include "simd/kernels.h"

namespace columnstore {

namespace {

template <bool LessThan>
std::size_t filter_int32_scalar_impl(const int32_t* data,
                                     std::size_t n,
                                     int32_t threshold,
                                     uint8_t* out_bitmap) {
    std::memset(out_bitmap, 0, (n + 7) / 8);
    std::size_t set_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool pass = LessThan ? (data[i] < threshold) : (data[i] > threshold);
        if (pass) {
            out_bitmap[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            ++set_count;
        }
    }
    return set_count;
}

} // namespace

std::size_t
filter_int32_lt_scalar(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap) {
    return filter_int32_scalar_impl<true>(data, n, threshold, out_bitmap);
}

std::size_t
filter_int32_gt_scalar(const int32_t* data, std::size_t n, int32_t threshold, uint8_t* out_bitmap) {
    return filter_int32_scalar_impl<false>(data, n, threshold, out_bitmap);
}

} // namespace columnstore
