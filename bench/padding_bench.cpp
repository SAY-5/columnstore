// Cache-line padding study.
//
// Goal: measure whether forcing 64-byte alignment on Batch<int32_t> and a
// minimal Column<int32_t>-shaped struct moves the needle on a tight scan +
// filter loop. We compare a baseline layout (default alignment) against an
// `alignas(64)`-padded layout. Both walks operate on identical data and the
// same kernel.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "bench_common.h"
#include "core/batch.h"
#include "obs/histogram.h"
#include "simd/cpu_detect.h"
#include "simd/kernels.h"

using namespace columnstore;
using namespace columnstore::bench;

namespace {

struct ColShape {
    const int32_t* data;
    std::size_t size;
};

struct alignas(64) ColShapeAligned {
    const int32_t* data;
    std::size_t size;
    // pad out to 64 bytes so any subsequent struct lives on a fresh line
    std::byte pad[64 - sizeof(const int32_t*) - sizeof(std::size_t)];
};

template <typename Col>
uint64_t run_filter(const std::vector<Col>& cols, std::vector<uint8_t>& bm, int32_t thr) {
    clobber();
    const uint64_t t0 = now_ns();
    std::size_t bm_off = 0;
    for (const auto& c : cols) {
        filter_int32_gt_scalar(c.data, c.size, thr, bm.data() + bm_off);
        bm_off += (c.size + 7) / 8;
    }
    const uint64_t t1 = now_ns();
    sink(bm[0]);
    return t1 - t0;
}

} // namespace

int main() {
    const std::size_t rows_per_col = env_rows(1'000'000);
    const std::size_t ncols = 64;
    const int iters = env_iters(20);

    std::printf("padding_bench: simd-path=%s rows/col=%zu ncols=%zu iters=%d\n",
                simd_path_name(active_simd_path()),
                rows_per_col,
                ncols,
                iters);

    // One shared payload reused via pointer aliasing so we measure metadata
    // layout effects, not data-cache footprint.
    auto payload = synth_int32(rows_per_col, 0xCAFE2026u);
    std::vector<uint8_t> bm(((rows_per_col + 7) / 8) * ncols, 0);

    std::vector<ColShape> base(ncols);
    std::vector<ColShapeAligned> aligned(ncols);
    for (std::size_t i = 0; i < ncols; ++i) {
        base[i].data = payload.data();
        base[i].size = rows_per_col;
        aligned[i].data = payload.data();
        aligned[i].size = rows_per_col;
    }

    LatencyHistogram h_base;
    LatencyHistogram h_align;
    // Warmup.
    (void)run_filter(base, bm, 50'000);
    (void)run_filter(aligned, bm, 50'000);

    for (int i = 0; i < iters; ++i) {
        h_base.add(run_filter(base, bm, 50'000));
    }
    for (int i = 0; i < iters; ++i) {
        h_align.add(run_filter(aligned, bm, 50'000));
    }

    auto report = [&](const char* label, LatencyHistogram& h, std::size_t struct_sz) {
        const double best = static_cast<double>(h.min()) / 1e9;
        const std::size_t total_rows = rows_per_col * ncols;
        const double tput = static_cast<double>(total_rows) / best;
        std::printf("  %-22s sizeof=%3zu  best=%.3f ms  %.3f B v/s  "
                    "p50=%llu p95=%llu p99=%llu ns\n",
                    label,
                    struct_sz,
                    best * 1e3,
                    tput / 1e9,
                    static_cast<unsigned long long>(h.percentile(50)),
                    static_cast<unsigned long long>(h.percentile(95)),
                    static_cast<unsigned long long>(h.percentile(99)));
    };
    std::printf("padding_bench results:\n");
    report("default-aligned col", h_base, sizeof(ColShape));
    report("64B-aligned col", h_align, sizeof(ColShapeAligned));

    if (const char* path = env_json_out()) {
        FILE* f = std::fopen(path, "a");
        if (f) {
            auto emit = [&](const char* op_label, LatencyHistogram& h, std::size_t sz) {
                const double best = static_cast<double>(h.min()) / 1e9;
                const std::size_t total_rows = rows_per_col * ncols;
                const double tput = static_cast<double>(total_rows) / best;
                std::fprintf(f,
                             "{\"bench\":\"padding\",\"op\":\"%s\",\"simd\":\"%s\","
                             "\"rows\":%zu,\"iters\":%d,\"sizeof_col\":%zu,"
                             "\"best_ns\":%llu,\"p50_ns\":%llu,\"p95_ns\":%llu,"
                             "\"p99_ns\":%llu,\"throughput_b_v_s\":%.6f,\"label\":\"%s\"}\n",
                             op_label,
                             simd_path_name(active_simd_path()),
                             total_rows,
                             iters,
                             sz,
                             static_cast<unsigned long long>(h.min()),
                             static_cast<unsigned long long>(h.percentile(50)),
                             static_cast<unsigned long long>(h.percentile(95)),
                             static_cast<unsigned long long>(h.percentile(99)),
                             tput / 1e9,
                             env_bench_label("default"));
                (void)best;
            };
            emit("padding_default", h_base, sizeof(ColShape));
            emit("padding_64b", h_align, sizeof(ColShapeAligned));
            std::fclose(f);
        }
    }
    return 0;
}
