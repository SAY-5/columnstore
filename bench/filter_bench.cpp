#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench_common.h"
#include "core/batch.h"
#include "obs/histogram.h"
#include "simd/cpu_detect.h"
#include "simd/kernels.h"

using namespace columnstore;
using namespace columnstore::bench;

int main() {
    const std::size_t rows = env_rows(100'000'000);
    const int iters = env_iters(5);

    std::printf("filter_bench: simd-path=%s rows=%zu iters=%d\n",
                simd_path_name(active_simd_path()),
                rows,
                iters);

    auto data = synth_int32(rows, 0xC01D5EEDu);
    std::vector<uint8_t> bm(Batch<int32_t>::bitmap_bytes(rows), 0);

    auto run_once = [&](bool use_avx2) -> uint64_t {
        clobber();
        const uint64_t t0 = now_ns();
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (use_avx2 && active_simd_path() == SimdPath::Avx2) {
            filter_int32_gt_avx2(data.data(), rows, 1000, bm.data());
        } else {
            filter_int32_gt_scalar(data.data(), rows, 1000, bm.data());
        }
#else
        (void)use_avx2;
        filter_int32_gt_scalar(data.data(), rows, 1000, bm.data());
#endif
        const uint64_t t1 = now_ns();
        sink(bm[0]);
        return t1 - t0;
    };

    // Warmup.
    run_once(true);
    run_once(false);

    LatencyHistogram h_avx;
    LatencyHistogram h_sc;
    for (int i = 0; i < iters; ++i) {
        h_avx.add(run_once(true));
    }
    for (int i = 0; i < iters; ++i) {
        h_sc.add(run_once(false));
    }

    auto report = [&](const char* label, LatencyHistogram& h) {
        const double best = static_cast<double>(h.min()) / 1e9;
        const double tput = static_cast<double>(rows) / best;
        std::printf("  %-8s best=%.3f ms  %.3f B v/s  p50=%llu p95=%llu p99=%llu ns\n",
                    label,
                    best * 1e3,
                    tput / 1e9,
                    static_cast<unsigned long long>(h.percentile(50)),
                    static_cast<unsigned long long>(h.percentile(95)),
                    static_cast<unsigned long long>(h.percentile(99)));
    };
    std::printf("filter_bench results:\n");
    report("avx2", h_avx);
    report("scalar", h_sc);

    if (const char* path = env_json_out()) {
        FILE* f = std::fopen(path, "a");
        if (f) {
            auto emit = [&](const char* op_label, LatencyHistogram& h) {
                const double best = static_cast<double>(h.min()) / 1e9;
                const double tput = static_cast<double>(rows) / best;
                std::fprintf(f,
                             "{\"bench\":\"filter\",\"op\":\"filter_gt_%s\",\"simd\":\"%s\","
                             "\"rows\":%zu,\"iters\":%d,\"best_ns\":%llu,\"p50_ns\":%llu,"
                             "\"p95_ns\":%llu,\"p99_ns\":%llu,\"throughput_b_v_s\":%.6f,"
                             "\"label\":\"%s\"}\n",
                             op_label,
                             simd_path_name(active_simd_path()),
                             rows,
                             iters,
                             static_cast<unsigned long long>(h.min()),
                             static_cast<unsigned long long>(h.percentile(50)),
                             static_cast<unsigned long long>(h.percentile(95)),
                             static_cast<unsigned long long>(h.percentile(99)),
                             tput / 1e9,
                             env_bench_label("default"));
                (void)best;
            };
            emit("avx2", h_avx);
            emit("scalar", h_sc);
            std::fclose(f);
        }
    }
    return 0;
}
