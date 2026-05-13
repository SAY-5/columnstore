#pragma once

#include <string>
#include <vector>

#include "core/types.h"

namespace columnstore {

struct ColumnDef {
    std::string name;
    DataType type;
};

class Schema {
public:
    Schema() = default;
    explicit Schema(std::vector<ColumnDef> cols) : cols_(std::move(cols)) {}

    void add_column(std::string name, DataType type) { cols_.push_back({std::move(name), type}); }

    std::size_t size() const { return cols_.size(); }
    const ColumnDef& at(std::size_t i) const { return cols_[i]; }
    const std::vector<ColumnDef>& columns() const { return cols_; }

    // Returns -1 if not found.
    int index_of(const std::string& name) const;

private:
    std::vector<ColumnDef> cols_;
};

} // namespace columnstore
