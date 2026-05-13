#include "rle/encoder.h"

namespace columnstore {

RleEncodedInt32 rle_encode_int32(const int32_t* data, std::size_t n) {
    RleEncodedInt32 out;
    if (n == 0) {
        return out;
    }
    int32_t current = data[0];
    uint32_t run_len = 1;
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] == current) {
            // Saturating run length. Pathological case only; with 100M-row
            // columns we never hit this, but guard regardless.
            if (run_len == UINT32_MAX) {
                out.values.push_back(current);
                out.lengths.push_back(run_len);
                run_len = 0;
            }
            ++run_len;
        } else {
            out.values.push_back(current);
            out.lengths.push_back(run_len);
            current = data[i];
            run_len = 1;
        }
    }
    out.values.push_back(current);
    out.lengths.push_back(run_len);
    return out;
}

Column<int32_t> rle_encode_column(const std::vector<int32_t>& data) {
    auto enc = rle_encode_int32(data.data(), data.size());
    return Column<int32_t>::from_rle(std::move(enc.values), std::move(enc.lengths));
}

} // namespace columnstore
