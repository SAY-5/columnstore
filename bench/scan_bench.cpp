#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench_common.h"
#include "exec/pipeline.h"
#include "obs/histogram.h"
#include "simd/cpu_detect.h"

using namespace columnstore;
using namespace columnstore::bench;

namespace {
void emit_json(const char* path,
               std::size_t rows,
               int iters,
               double best_secs,
               uint64_t p50,
               uint64_t p95,
               uint64_t p99) {
    if (!path) {
        return;
    }
    FILE* f = std::fopen(path, "a");
    if (!f) {
        return;
    }
    const double tput = static_cast<double>(rows) / best_secs;
    std::fprintf(f,
                 "{\"bench\":\"scan\",\"op\":\"scan\",\"simd\":\"%s\",\"rows\":%zu,"
                 "\"iters\":%d,\"best_ns\":%llu,\"p50_ns\":%llu,\"p95_ns\":%llu,"
                 "\"p99_ns\":%llu,\"throughput_b_v_s\":%.6f,\"label\":\"%s\"}\n",
                 simd_path_name(active_simd_path()),
                 rows,
                 iters,
                 static_cast<unsigned long long>(best_secs * 1e9),
                 static_cast<unsigned long long>(p50),
                 static_cast<unsigned long long>(p95),
                 static_cast<unsigned long long>(p99),
                 tput / 1e9,
                 env_bench_label("default"));
    std::fclose(f);
}
} // namespace

int main() {
    const std::size_t rows = env_rows(100'000'000);
    const int iters = env_iters(5);

    std::printf("scan_bench: simd-path=%s rows=%zu iters=%d\n",
                simd_path_name(active_simd_path()),
                rows,
                iters);

    auto data = synth_int32(rows, 0xC01D5EEDu);

    // Warmup.
    {
        PipelineSpec spec;
        spec.filter_op = FilterOp::GreaterThan;
        spec.threshold = 1000;
        spec.agg = AggregateKind::Sum;
        auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
        sink(r.sum);
    }

    LatencyHistogram h;
    int64_t last_sum = 0;
    for (int i = 0; i < iters; ++i) {
        clobber();
        const uint64_t t0 = now_ns();
        PipelineSpec spec;
        spec.filter_op = FilterOp::GreaterThan;
        spec.threshold = 1000;
        spec.agg = AggregateKind::Sum;
        auto r = PipelineBuilder::run_raw(data.data(), data.size(), spec);
        const uint64_t t1 = now_ns();
        h.add(t1 - t0);
        last_sum = r.sum;
        sink(last_sum);
    }

    const uint64_t p50 = h.percentile(50);
    const uint64_t p95 = h.percentile(95);
    const uint64_t p99 = h.percentile(99);
    const double best_secs = static_cast<double>(h.min()) / 1e9;
    const double values_per_sec = static_cast<double>(rows) / best_secs;

    std::printf("scan_bench results:\n");
    std::printf("  rows                  : %zu\n", rows);
    std::printf("  best wall-clock       : %.3f ms\n", best_secs * 1e3);
    std::printf("  throughput (best)     : %.3f M values/sec\n", values_per_sec / 1e6);
    std::printf("  throughput (best)     : %.3f B values/sec\n", values_per_sec / 1e9);
    std::printf("  p50 / p95 / p99 ns    : %llu / %llu / %llu\n",
                static_cast<unsigned long long>(p50),
                static_cast<unsigned long long>(p95),
                static_cast<unsigned long long>(p99));
    std::printf("  result sum            : %lld\n", static_cast<long long>(last_sum));

    emit_json(env_json_out(), rows, iters, best_secs, p50, p95, p99);
    return 0;
}
