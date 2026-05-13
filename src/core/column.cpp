#include "core/column.h"

namespace columnstore {

template <typename T> std::vector<T> Column<T>::materialize() const {
    if (!has_rle_) {
        return raw_;
    }
    std::vector<T> out;
    out.reserve(rle_row_count_);
    for (std::size_t i = 0; i < rle_values_.size(); ++i) {
        const T v = rle_values_[i];
        const uint32_t len = rle_lengths_[i];
        for (uint32_t k = 0; k < len; ++k) {
            out.push_back(v);
        }
    }
    return out;
}

template class Column<int32_t>;
template class Column<int64_t>;
template class Column<double>;

} // namespace columnstore
