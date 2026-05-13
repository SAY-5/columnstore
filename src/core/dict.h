#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace columnstore {

// Dictionary-encoded column for low-cardinality types.
//
// Storage:
//   dict[k]        -> the k-th distinct value, in insertion order
//   codes[i]       -> the dictionary index of row i (uint8_t if dict.size() <= 256,
//                     uint16_t up to 65536)
//
// Build-time check: dict.size() <= 256 lets us use the byte-wide codes buffer
// and SIMD-accelerate equality filters via _mm256_cmpeq_epi8.
template <typename T> class DictColumn {
public:
    static constexpr std::size_t kMaxCardinalityForByteCodes = 256;

    static std::optional<DictColumn<T>>
    try_build(const std::vector<T>& raw, std::size_t max_card = kMaxCardinalityForByteCodes) {
        DictColumn<T> d;
        std::unordered_map<T, std::size_t> seen;
        seen.reserve(max_card);
        d.codes_.reserve(raw.size());
        for (const T& v : raw) {
            auto it = seen.find(v);
            std::size_t idx;
            if (it == seen.end()) {
                idx = d.dict_.size();
                if (idx >= max_card) {
                    return std::nullopt;
                }
                d.dict_.push_back(v);
                seen.emplace(v, idx);
            } else {
                idx = it->second;
            }
            d.codes_.push_back(static_cast<uint8_t>(idx));
        }
        return d;
    }

    std::size_t row_count() const { return codes_.size(); }
    std::size_t cardinality() const { return dict_.size(); }
    const std::vector<T>& dict() const { return dict_; }
    const std::vector<uint8_t>& codes() const { return codes_; }
    const uint8_t* codes_data() const { return codes_.data(); }

    // Look up a value in the dictionary. Returns the dict index, or
    // std::nullopt if not present.
    std::optional<uint8_t> code_for(const T& v) const {
        for (std::size_t i = 0; i < dict_.size(); ++i) {
            if (dict_[i] == v) {
                return static_cast<uint8_t>(i);
            }
        }
        return std::nullopt;
    }

    // Materialize back to a raw vector for compatibility with code that
    // expects a flat column.
    std::vector<T> materialize() const {
        std::vector<T> out;
        out.reserve(codes_.size());
        for (uint8_t c : codes_) {
            out.push_back(dict_[c]);
        }
        return out;
    }

private:
    std::vector<T> dict_;
    std::vector<uint8_t> codes_;
};

extern template class DictColumn<int32_t>;
extern template class DictColumn<std::string>;

} // namespace columnstore
