#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "exec/pipeline.h"
#include "rle/encoder.h"
#include "simd/cpu_detect.h"

using namespace columnstore;

namespace {

void usage() {
    std::printf("usage: columnstore [--rows N] [--threshold T] [--seed S] [--rle]\n"
                "  --rows N        number of int32 rows (default 1000000)\n"
                "  --threshold T   filter threshold; query is SUM(c) WHERE c > T (default 1000)\n"
                "  --seed S        rng seed (default 42)\n"
                "  --rle           encode the column with RLE before scanning\n");
}

int64_t reference_sum(const std::vector<int32_t>& data, int32_t threshold) {
    int64_t s = 0;
    for (auto v : data) {
        if (v > threshold) {
            s += static_cast<int64_t>(v);
        }
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    std::size_t rows = 1'000'000;
    int32_t threshold = 1000;
    uint64_t seed = 42;
    bool use_rle = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else if (arg == "--rows" && i + 1 < argc) {
            rows = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--rle") {
            use_rle = true;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            usage();
            return 2;
        }
    }

    std::printf("columnstore: simd-path=%s rows=%zu threshold=%d rle=%s\n",
                simd_path_name(active_simd_path()),
                rows,
                threshold,
                use_rle ? "yes" : "no");

    // Synthesize a deterministic int32 column.
    std::vector<int32_t> data;
    data.reserve(rows);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, 100'000);
    for (std::size_t i = 0; i < rows; ++i) {
        data.push_back(dist(rng));
    }

    PipelineSpec spec;
    spec.filter_op = FilterOp::GreaterThan;
    spec.threshold = threshold;
    spec.agg = AggregateKind::Sum;

    AggregateResult r;
    auto t0 = std::chrono::steady_clock::now();
    if (use_rle) {
        const Column<int32_t> col = rle_encode_column(data);
        r = PipelineBuilder::run_column(col, spec);
    } else {
        r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
    }
    auto t1 = std::chrono::steady_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    const int64_t ref = reference_sum(data, threshold);
    const bool ok = (r.sum == ref);
    const double secs = static_cast<double>(ns) / 1e9;
    const double values_per_sec = static_cast<double>(rows) / secs;

    std::printf("result: sum=%lld count=%zu ok=%s\n",
                static_cast<long long>(r.sum),
                r.count,
                ok ? "yes" : "no");
    std::printf("perf:   %.3f ms   %.3f M values/sec\n", secs * 1e3, values_per_sec / 1e6);

    return ok ? 0 : 1;
}
