#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "exec/pipeline.h"
#include "rle/encoder.h"
#include "simd/cpu_detect.h"

using namespace columnstore;

namespace {

std::size_t integration_rows() {
    if (const char* env = std::getenv("COLUMNSTORE_INTEGRATION_ROWS")) {
        return static_cast<std::size_t>(std::stoull(env));
    }
    // Default: 5M rows. CI overrides via env to 1M for speed. The README
    // bench numbers come from `scan_bench` at 100M.
    return 5'000'000;
}

} // namespace

TEST(QueryIntegration, SumGreaterThanMatchesReference) {
    const std::size_t n = integration_rows();
    std::vector<int32_t> data(n);
    std::mt19937_64 rng(0xFEEDFACEC0FFEEull);
    std::uniform_int_distribution<int32_t> dist(0, 10'000);
    for (auto& v : data) {
        v = dist(rng);
    }

    int64_t reference = 0;
    std::size_t reference_count = 0;
    for (auto v : data) {
        if (v > 1000) {
            reference += v;
            ++reference_count;
        }
    }

    PipelineSpec spec;
    spec.filter_op = FilterOp::GreaterThan;
    spec.threshold = 1000;
    spec.agg = AggregateKind::Sum;
    auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);

    EXPECT_EQ(r.sum, reference);
    EXPECT_EQ(r.count, reference_count);
    SUCCEED() << "simd-path=" << simd_path_name(active_simd_path()) << " rows=" << n;
}

TEST(QueryIntegration, RleAndRawAgree) {
    const std::size_t n = integration_rows();
    // Construct a moderately compressible column: each value repeats 8x on
    // average.
    std::vector<int32_t> data;
    data.reserve(n);
    std::mt19937_64 rng(0xC0FFEEAAAAul);
    while (data.size() < n) {
        const int32_t v = static_cast<int32_t>(rng() % 5000);
        const std::size_t len = 1 + (rng() % 16);
        for (std::size_t k = 0; k < len && data.size() < n; ++k) {
            data.push_back(v);
        }
    }

    PipelineSpec spec;
    spec.filter_op = FilterOp::GreaterThan;
    spec.threshold = 2500;
    spec.agg = AggregateKind::Sum;

    auto raw = PipelineBuilder::run_raw(data.data(), data.size(), spec);
    Column<int32_t> rle = rle_encode_column(data);
    auto from_rle = PipelineBuilder::run_column(rle, spec);

    EXPECT_EQ(raw.sum, from_rle.sum);
    EXPECT_EQ(raw.count, from_rle.count);
}
