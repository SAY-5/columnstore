#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/types.h"

namespace columnstore {

// A typed contiguous column. Storage is either raw (a flat array of values)
// or RLE-compressed (a parallel pair of value/length runs).
template <typename T> class Column {
public:
    Column() = default;

    static Column<T> from_raw(std::vector<T> data) {
        Column<T> c;
        c.raw_ = std::move(data);
        c.has_rle_ = false;
        return c;
    }

    static Column<T> from_rle(std::vector<T> values, std::vector<uint32_t> lengths) {
        Column<T> c;
        c.rle_values_ = std::move(values);
        c.rle_lengths_ = std::move(lengths);
        c.has_rle_ = true;
        // logical row count is the sum of run lengths
        std::size_t total = 0;
        for (auto l : c.rle_lengths_) {
            total += l;
        }
        c.rle_row_count_ = total;
        return c;
    }

    bool is_rle() const { return has_rle_; }
    std::size_t row_count() const { return has_rle_ ? rle_row_count_ : raw_.size(); }

    const T* raw_data() const { return raw_.data(); }
    T* raw_data_mut() { return raw_.data(); }
    const std::vector<T>& raw() const { return raw_; }
    std::vector<T>& raw_mut() { return raw_; }

    const std::vector<T>& rle_values() const { return rle_values_; }
    const std::vector<uint32_t>& rle_lengths() const { return rle_lengths_; }

    // Materialize this column to a flat raw buffer, decoding RLE if needed.
    std::vector<T> materialize() const;

private:
    std::vector<T> raw_;
    std::vector<T> rle_values_;
    std::vector<uint32_t> rle_lengths_;
    std::size_t rle_row_count_ = 0;
    bool has_rle_ = false;
};

// Variable-length string column. Strings are stored back-to-back in `arena`
// with `offsets[i]..offsets[i+1]` defining the byte range of row i.
class StringColumn {
public:
    StringColumn() { offsets_.push_back(0); }

    void append(const std::string& s) {
        arena_.insert(arena_.end(), s.begin(), s.end());
        offsets_.push_back(static_cast<uint32_t>(arena_.size()));
    }

    std::size_t row_count() const { return offsets_.size() - 1; }

    std::string_view at(std::size_t i) const {
        const char* base = reinterpret_cast<const char*>(arena_.data());
        return std::string_view(base + offsets_[i], offsets_[i + 1] - offsets_[i]);
    }

    const std::vector<uint8_t>& arena() const { return arena_; }
    const std::vector<uint32_t>& offsets() const { return offsets_; }

private:
    std::vector<uint8_t> arena_;
    std::vector<uint32_t> offsets_;
};

// Explicit instantiations live in column.cpp.
extern template class Column<int32_t>;
extern template class Column<int64_t>;
extern template class Column<double>;

} // namespace columnstore
