#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "exec/operator.h"

namespace columnstore {

enum class AggregateKind {
    Sum = 0,
    Count = 1,
    Min = 2,
    Max = 3,
    Avg = 4,
};

struct AggregateResult {
    AggregateKind kind = AggregateKind::Sum;
    int64_t sum = 0;
    std::size_t count = 0;
    int32_t min = 0;
    int32_t max = 0;
    bool empty = true;

    double avg() const {
        return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count);
    }
};

// Pulls batches from the child operator, applying the selection bitmap (if
// present) and folding into an AggregateResult.
class Int32Aggregate {
public:
    Int32Aggregate(std::unique_ptr<Int32Operator> child, AggregateKind kind);

    AggregateResult run();

private:
    std::unique_ptr<Int32Operator> child_;
    AggregateKind kind_;
};

} // namespace columnstore
