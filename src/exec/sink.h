#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "exec/operator.h"

namespace columnstore {

// Materializes the results of a pipeline into a flat vector. Only rows whose
// selection bit is set are kept. Useful for tests; not used on the bench hot
// path.
class Int32Sink {
public:
    explicit Int32Sink(std::unique_ptr<Int32Operator> child);

    std::vector<int32_t> collect();

private:
    std::unique_ptr<Int32Operator> child_;
};

} // namespace columnstore
