#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace columnstore {

enum class DataType : uint8_t {
    Int32 = 0,
    Int64 = 1,
    Double = 2,
    String = 3,
};

inline const char* data_type_name(DataType t) {
    switch (t) {
    case DataType::Int32:
        return "int32";
    case DataType::Int64:
        return "int64";
    case DataType::Double:
        return "double";
    case DataType::String:
        return "string";
    }
    return "unknown";
}

using Value = std::variant<int32_t, int64_t, double, std::string>;

// Vector-at-a-time batch size. Chosen to keep a batch comfortably in L1d on
// modern x86_64 cores: 4096 * sizeof(int32) = 16 KiB.
constexpr std::size_t kBatchSize = 4096;

} // namespace columnstore
