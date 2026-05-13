#pragma once

#include <memory>

#include "exec/operator.h"

namespace columnstore {

enum class FilterOp {
    LessThan = 0,
    GreaterThan = 1,
};

// `data[i] < threshold` (or `>`) per row. Forwards batches downstream with a
// populated `selection` bitmap. The kernel path (AVX2 vs scalar) is chosen
// at construction from active_simd_path().
class Int32Filter : public Int32Operator {
public:
    Int32Filter(std::unique_ptr<Int32Operator> child, FilterOp op, int32_t threshold);

    std::optional<Batch<int32_t>> next() override;

    // For tests + reporting.
    std::size_t rows_passed() const { return rows_passed_; }
    std::size_t rows_seen() const { return rows_seen_; }

private:
    std::unique_ptr<Int32Operator> child_;
    FilterOp op_;
    int32_t threshold_;
    std::size_t rows_passed_ = 0;
    std::size_t rows_seen_ = 0;
};

} // namespace columnstore
