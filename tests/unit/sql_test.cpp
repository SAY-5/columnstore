#include <cstdint>
#include <random>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "io/csv.h"
#include "sql/sql.h"

using namespace columnstore;
using namespace columnstore::sql;

namespace {

std::string make_csv_100k(int /*seed*/) {
    // 100k rows. Columns: id (int32), bucket (int32, cardinality 4),
    // amount (double). Bucket cycles 0..3; amount is 1.0 per row so
    // hand-computed expected results are exact integers. Deterministic by
    // construction; the seed parameter is kept for future fuzz variants.
    std::ostringstream os;
    os << "# types: int32, int32, double\n";
    os << "id, bucket, amount\n";
    for (int i = 0; i < 100'000; ++i) {
        const int bucket = i % 4;
        os << i << ", " << bucket << ", 1.0\n";
    }
    return os.str();
}

} // namespace

TEST(Sql, ParseSelectAggWithWhereAndGroupBy) {
    Query q = parse("SELECT bucket, SUM(amount) FROM t WHERE bucket > 0 GROUP BY bucket");
    EXPECT_EQ(q.projections.size(), 2u);
    EXPECT_EQ(q.projections[0].agg, AggKind::None);
    EXPECT_EQ(q.projections[0].col, "bucket");
    EXPECT_EQ(q.projections[1].agg, AggKind::Sum);
    EXPECT_EQ(q.projections[1].col, "amount");
    EXPECT_EQ(q.table, "t");
    ASSERT_TRUE(q.where.present);
    EXPECT_EQ(q.where.col, "bucket");
    EXPECT_EQ(q.where.op, CmpOp::Gt);
    ASSERT_TRUE(std::holds_alternative<int64_t>(q.where.value));
    EXPECT_EQ(std::get<int64_t>(q.where.value), 0);
    EXPECT_EQ(q.group_by, "bucket");
}

TEST(Sql, RejectsInvalidInput) {
    EXPECT_THROW(parse("SELECT FROM t"), std::runtime_error);
    EXPECT_THROW(parse("SELECT a t"), std::runtime_error);
    EXPECT_THROW(parse("SELECT a FROM t WHERE x"), std::runtime_error);
}

TEST(Sql, Execute100kCsvSumWithFilter) {
    const std::string csv = make_csv_100k(1);
    auto table = io::read_csv_string(csv);
    ASSERT_EQ(table.row_count, 100'000u);
    table.cols[0].name = table.cols[0].name; // ensure stable
    Query q = parse("SELECT SUM(amount) FROM t WHERE bucket > 0");
    auto rs = execute(q, table);
    ASSERT_EQ(rs.rows.size(), 1u);
    ASSERT_EQ(rs.rows[0].cells.size(), 1u);
    // 75,000 rows have bucket in {1,2,3}; each contributes amount=1.0.
    // SUM over a double column returns a double cell.
    ASSERT_TRUE(std::holds_alternative<double>(rs.rows[0].cells[0]));
    EXPECT_DOUBLE_EQ(std::get<double>(rs.rows[0].cells[0]), 75'000.0);
}

TEST(Sql, Execute100kCsvGroupBySum) {
    const std::string csv = make_csv_100k(2);
    auto table = io::read_csv_string(csv);
    Query q = parse("SELECT bucket, SUM(amount) FROM t GROUP BY bucket");
    auto rs = execute(q, table);
    ASSERT_EQ(rs.rows.size(), 4u);
    // Map (bucket -> sum) for stable assertion under map ordering.
    std::map<int64_t, double> got;
    for (const auto& r : rs.rows) {
        const auto k = std::get<int64_t>(r.cells[0]);
        const auto v = std::get<double>(r.cells[1]);
        got[k] = v;
    }
    for (int b = 0; b < 4; ++b) {
        EXPECT_DOUBLE_EQ(got[b], 25'000.0) << "bucket " << b;
    }
}

TEST(Sql, ExecuteCountStarLike) {
    const std::string csv = make_csv_100k(3);
    auto table = io::read_csv_string(csv);
    // Use COUNT(id) as a stand-in for COUNT(*) since this grammar requires
    // a column.
    Query q = parse("SELECT COUNT(id) FROM t WHERE bucket = 2");
    auto rs = execute(q, table);
    ASSERT_EQ(rs.rows.size(), 1u);
    EXPECT_EQ(std::get<int64_t>(rs.rows[0].cells[0]), 25'000);
}

TEST(Sql, ExecuteCountDistinct) {
    const std::string csv = make_csv_100k(4);
    auto table = io::read_csv_string(csv);
    Query q = parse("SELECT COUNT_DISTINCT(bucket) FROM t");
    auto rs = execute(q, table);
    ASSERT_EQ(rs.rows.size(), 1u);
    EXPECT_EQ(std::get<int64_t>(rs.rows[0].cells[0]), 4);
}
