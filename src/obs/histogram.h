#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace columnstore {

// Minimal latency histogram. Stores raw samples; computes percentiles by
// sorting. Plenty fast for the bench harness (a few thousand samples per
// run).
class LatencyHistogram {
public:
    void add(uint64_t ns) { samples_.push_back(ns); }
    std::size_t count() const { return samples_.size(); }

    // P in [0, 100]. Returns 0 if empty.
    uint64_t percentile(double p) const;

    uint64_t min() const;
    uint64_t max() const;
    double mean() const;

private:
    mutable std::vector<uint64_t> samples_;
    mutable bool sorted_ = false;
    void sort_lazy() const;
};

} // namespace columnstore
