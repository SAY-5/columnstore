#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "exec/pipeline.h"
#include "io/csv.h"
#include "rle/encoder.h"
#include "simd/cpu_detect.h"
#include "sql/sql.h"

using namespace columnstore;

namespace {

void usage() {
    std::printf("usage:\n"
                "  columnstore [--rows N] [--threshold T] [--seed S] [--rle]\n"
                "      built-in benchmark query (default mode).\n"
                "  columnstore --sql FILE \"QUERY\"\n"
                "      load CSV from FILE (with `# types:` header) and run a SQL query\n"
                "      against table 't'. The grammar covers:\n"
                "          SELECT col1, AGG(col2) FROM t\n"
                "            [WHERE col3 op literal]\n"
                "            [GROUP BY col1]\n"
                "      where AGG is one of SUM, COUNT, MIN, MAX, AVG, COUNT_DISTINCT.\n");
}

int run_sql(const char* csv_path, const char* query) {
    auto table = io::read_csv(csv_path);
    sql::Query q = sql::parse(query);
    if (q.table != "t") {
        std::fprintf(stderr, "sql: table name must be 't' (got '%s')\n", q.table.c_str());
        return 2;
    }
    auto rs = sql::execute(q, table);
    for (std::size_t i = 0; i < rs.column_names.size(); ++i) {
        std::printf("%s%s", i == 0 ? "" : "\t", rs.column_names[i].c_str());
    }
    std::printf("\n");
    for (const auto& row : rs.rows) {
        for (std::size_t i = 0; i < row.cells.size(); ++i) {
            const auto& c = row.cells[i];
            if (std::holds_alternative<int64_t>(c)) {
                std::printf(
                    "%s%lld", i == 0 ? "" : "\t", static_cast<long long>(std::get<int64_t>(c)));
            } else if (std::holds_alternative<double>(c)) {
                std::printf("%s%.6g", i == 0 ? "" : "\t", std::get<double>(c));
            } else {
                std::printf("%s%s", i == 0 ? "" : "\t", std::get<std::string>(c).c_str());
            }
        }
        std::printf("\n");
    }
    return 0;
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
        } else if (arg == "--sql" && i + 2 < argc) {
            return run_sql(argv[i + 1], argv[i + 2]);
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
