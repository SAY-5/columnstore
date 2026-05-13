#include "exec/pipeline.h"

#include "exec/source.h"

namespace columnstore {

AggregateResult
PipelineBuilder::run_raw(const int32_t* data, std::size_t n, const PipelineSpec& spec) {
    auto scan = std::make_unique<Int32ColumnScan>(data, n);
    auto filt = std::make_unique<Int32Filter>(std::move(scan), spec.filter_op, spec.threshold);
    Int32Aggregate agg(std::move(filt), spec.agg);
    return agg.run();
}

AggregateResult PipelineBuilder::run_column(const Column<int32_t>& col, const PipelineSpec& spec) {
    std::unique_ptr<Int32Operator> scan;
    if (col.is_rle()) {
        scan = std::make_unique<Int32ColumnScan>(
            col.rle_values().data(), col.rle_lengths().data(), col.rle_values().size());
    } else {
        scan = std::make_unique<Int32ColumnScan>(col.raw_data(), col.row_count());
    }
    auto filt = std::make_unique<Int32Filter>(std::move(scan), spec.filter_op, spec.threshold);
    Int32Aggregate agg(std::move(filt), spec.agg);
    return agg.run();
}

} // namespace columnstore
