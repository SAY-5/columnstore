#include "obs/histogram.h"

namespace columnstore {

void LatencyHistogram::sort_lazy() const {
    if (!sorted_) {
        std::sort(samples_.begin(), samples_.end());
        sorted_ = true;
    }
}

uint64_t LatencyHistogram::percentile(double p) const {
    if (samples_.empty()) {
        return 0;
    }
    sort_lazy();
    if (p <= 0.0) {
        return samples_.front();
    }
    if (p >= 100.0) {
        return samples_.back();
    }
    const double idx = (p / 100.0) * static_cast<double>(samples_.size() - 1);
    const std::size_t i = static_cast<std::size_t>(idx);
    return samples_[i];
}

uint64_t LatencyHistogram::min() const {
    if (samples_.empty()) {
        return 0;
    }
    sort_lazy();
    return samples_.front();
}

uint64_t LatencyHistogram::max() const {
    if (samples_.empty()) {
        return 0;
    }
    sort_lazy();
    return samples_.back();
}

double LatencyHistogram::mean() const {
    if (samples_.empty()) {
        return 0.0;
    }
    long double total = 0.0L;
    for (uint64_t v : samples_) {
        total += static_cast<long double>(v);
    }
    return static_cast<double>(total / static_cast<long double>(samples_.size()));
}

} // namespace columnstore
