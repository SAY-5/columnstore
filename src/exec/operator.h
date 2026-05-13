#pragma once

#include <cstddef>
#include <optional>

#include "core/batch.h"
#include "core/types.h"

namespace columnstore {

// Base class for vector-at-a-time operators over int32 columns.
//
// `next()` returns the next batch (up to kBatchSize values). When the source
// is exhausted, returns std::nullopt. Operators are pull-based: a consumer
// calls `next()` on its child.
class Int32Operator {
public:
    virtual ~Int32Operator() = default;
    virtual std::optional<Batch<int32_t>> next() = 0;
};

} // namespace columnstore
