#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/column.h"
#include "exec/aggregate.h"
#include "exec/filter.h"
#include "exec/operator.h"

namespace columnstore {

// Convenience builder for the canonical query `SELECT <agg>(c) WHERE c <op> threshold`.
struct PipelineSpec {
    FilterOp filter_op = FilterOp::GreaterThan;
    int32_t threshold = 0;
    AggregateKind agg = AggregateKind::Sum;
};

class PipelineBuilder {
public:
    // Build over a raw int32 buffer.
    static AggregateResult run_raw(const int32_t* data, std::size_t n, const PipelineSpec& spec);

    // Build over a column (handles raw + RLE transparently).
    static AggregateResult run_column(const Column<int32_t>& col, const PipelineSpec& spec);
};

} // namespace columnstore
