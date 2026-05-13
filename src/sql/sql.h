#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "io/csv.h"

namespace columnstore {
namespace sql {

// Minimal SQL frontend.
//
// Grammar (single line, no joins, single aggregate):
//
//   query     := SELECT proj_list FROM ident
//                [WHERE ident op literal]
//                [GROUP BY ident]
//   proj_list := proj { , proj }
//   proj      := ident | agg ( ident )
//   agg       := SUM | COUNT | MIN | MAX | AVG | COUNT_DISTINCT
//   op        := = | != | < | <= | > | >=
//   literal   := number | string-literal
//
// The execution engine assumes a single table named `t` per query; the
// caller binds a CsvTable to that name and gets back a result.

enum class AggKind {
    None = 0,
    Sum,
    Count,
    Min,
    Max,
    Avg,
    CountDistinct,
};

enum class CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
};

struct Projection {
    std::string col; // input column
    AggKind agg = AggKind::None;
    std::string alias; // for output naming; defaults to col or "agg(col)"
};

struct WhereClause {
    bool present = false;
    std::string col;
    CmpOp op = CmpOp::Eq;
    // The literal is one of: int64, double, string.
    std::variant<int64_t, double, std::string> value;
};

struct Query {
    std::vector<Projection> projections;
    std::string table;
    WhereClause where;
    std::string group_by; // empty when absent
};

// Parse a single SQL statement. Throws std::runtime_error on bad input.
Query parse(const std::string& src);

// Cell value in an output row.
using Cell = std::variant<int64_t, double, std::string>;

struct ResultRow {
    std::vector<Cell> cells;
};

struct ResultSet {
    std::vector<std::string> column_names;
    std::vector<ResultRow> rows;
};

// Execute a parsed query against an in-memory CsvTable. The query's `table`
// field must equal the second argument's name.
ResultSet execute(const Query& q, const io::CsvTable& table);

} // namespace sql
} // namespace columnstore
