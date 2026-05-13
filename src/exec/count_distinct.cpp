#include "exec/count_distinct.h"

namespace columnstore {

std::size_t
filter_dict_eq_scalar(const uint8_t* codes, std::size_t n, uint8_t target, uint8_t* out_bm) {
    const std::size_t bytes = (n + 7) / 8;
    for (std::size_t i = 0; i < bytes; ++i) {
        out_bm[i] = 0;
    }
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (codes[i] == target) {
            out_bm[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            ++cnt;
        }
    }
    return cnt;
}

} // namespace columnstore
