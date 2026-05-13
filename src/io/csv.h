#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "core/column.h"

namespace columnstore {
namespace io {

enum class ColType {
    Int32,
    Int64,
    Double,
    String,
};

const char* col_type_name(ColType t);

// One column's storage after CSV ingest. Variant carries the actual buffer.
struct CsvColumn {
    std::string name;
    ColType type;
    std::vector<int32_t> i32;
    std::vector<int64_t> i64;
    std::vector<double> f64;
    std::vector<std::string> str;

    std::size_t row_count() const;
};

struct CsvTable {
    std::vector<CsvColumn> cols;
    std::size_t row_count = 0;

    const CsvColumn* find(const std::string& name) const {
        for (const auto& c : cols) {
            if (c.name == name) {
                return &c;
            }
        }
        return nullptr;
    }
};

// Read a CSV file with this format:
//
//     # types: int32, int32, string, double
//     id, age, name, score
//     1, 32, alice, 0.9
//     2, 41, bob, 0.7
//
// The first line (starting with `# types:`) is a type-hint comment that
// defines each column's type, in order. The second line is the header that
// defines column names. All subsequent lines are data rows.
//
// Throws std::runtime_error on parse failure.
CsvTable read_csv(const std::string& path);

// Read CSV from an in-memory string. Same format as read_csv().
CsvTable read_csv_string(const std::string& content);

} // namespace io
} // namespace columnstore
