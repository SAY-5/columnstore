#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "exec/pipeline.h"
#include "rle/encoder.h"

using namespace columnstore;

namespace {

int64_t reference_sum_gt(const std::vector<int32_t>& v, int32_t thr) {
    int64_t s = 0;
    for (auto x : v) {
        if (x > thr) {
            s += static_cast<int64_t>(x);
        }
    }
    return s;
}

std::size_t reference_count_gt(const std::vector<int32_t>& v, int32_t thr) {
    std::size_t c = 0;
    for (auto x : v) {
        if (x > thr) {
            ++c;
        }
    }
    return c;
}

} // namespace

TEST(Pipeline, SumGreaterThanRawColumn) {
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int32_t> dist(0, 10'000);
    std::vector<int32_t> data(50'000);
    for (auto& v : data) {
        v = dist(rng);
    }

    PipelineSpec spec;
    spec.filter_op = FilterOp::GreaterThan;
    spec.threshold = 1000;
    spec.agg = AggregateKind::Sum;

    auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
    EXPECT_EQ(r.sum, reference_sum_gt(data, 1000));
    EXPECT_EQ(r.count, reference_count_gt(data, 1000));
}

TEST(Pipeline, SumGreaterThanRleColumn) {
    // Highly compressible input.
    std::vector<int32_t> data;
    data.reserve(40'000);
    for (int v = 0; v < 100; ++v) {
        for (int k = 0; k < 400; ++k) {
            data.push_back(v * 10);
        }
    }
    Column<int32_t> col = rle_encode_column(data);
    EXPECT_TRUE(col.is_rle());

    PipelineSpec spec;
    spec.filter_op = FilterOp::GreaterThan;
    spec.threshold = 250;
    spec.agg = AggregateKind::Sum;

    auto r = PipelineBuilder::run_column(col, spec);
    EXPECT_EQ(r.sum, reference_sum_gt(data, 250));
    EXPECT_EQ(r.count, reference_count_gt(data, 250));
}

TEST(Pipeline, CountMinMaxOverRandom) {
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int32_t> dist(-500, 500);
    std::vector<int32_t> data(20'000);
    for (auto& v : data) {
        v = dist(rng);
    }
    {
        PipelineSpec spec;
        spec.filter_op = FilterOp::GreaterThan;
        spec.threshold = -1000;
        spec.agg = AggregateKind::Count;
        auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
        EXPECT_EQ(r.count, data.size());
    }
    {
        PipelineSpec spec;
        spec.filter_op = FilterOp::GreaterThan;
        spec.threshold = -1000;
        spec.agg = AggregateKind::Min;
        auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
        int32_t want = data[0];
        for (auto v : data) {
            if (v < want)
                want = v;
        }
        EXPECT_EQ(r.min, want);
    }
    {
        PipelineSpec spec;
        spec.filter_op = FilterOp::GreaterThan;
        spec.threshold = -1000;
        spec.agg = AggregateKind::Max;
        auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
        int32_t want = data[0];
        for (auto v : data) {
            if (v > want)
                want = v;
        }
        EXPECT_EQ(r.max, want);
    }
}

TEST(Pipeline, EmptyColumn) {
    PipelineSpec spec;
    spec.filter_op = FilterOp::GreaterThan;
    spec.threshold = 0;
    spec.agg = AggregateKind::Sum;
    auto r = PipelineBuilder::run_raw(nullptr, 0, spec);
    EXPECT_EQ(r.sum, 0);
    EXPECT_EQ(r.count, 0u);
    EXPECT_TRUE(r.empty);
}
