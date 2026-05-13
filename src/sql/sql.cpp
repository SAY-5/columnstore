#include "sql/sql.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace columnstore {
namespace sql {

namespace {

// ---------------------------- Lexer -------------------------------------

enum class TokKind {
    Ident,
    Number,
    StringLit,
    Op,
    End,
};

struct Tok {
    TokKind kind = TokKind::End;
    std::string text;
};

bool is_ident_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}
bool is_ident_cont(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::vector<Tok> tokenize(const std::string& src) {
    std::vector<Tok> out;
    std::size_t i = 0;
    while (i < src.size()) {
        const char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        if (is_ident_start(c)) {
            std::size_t j = i + 1;
            while (j < src.size() && is_ident_cont(src[j])) {
                ++j;
            }
            out.push_back({TokKind::Ident, src.substr(i, j - i)});
            i = j;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '-' && i + 1 < src.size() &&
             std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
            std::size_t j = i + (c == '-' ? 1 : 0);
            bool seen_dot = false;
            while (j < src.size() &&
                   (std::isdigit(static_cast<unsigned char>(src[j])) || src[j] == '.')) {
                if (src[j] == '.') {
                    if (seen_dot) {
                        break;
                    }
                    seen_dot = true;
                }
                ++j;
            }
            out.push_back({TokKind::Number, src.substr(i, j - i)});
            i = j;
            continue;
        }
        if (c == '\'') {
            std::size_t j = i + 1;
            std::string s;
            while (j < src.size() && src[j] != '\'') {
                s.push_back(src[j]);
                ++j;
            }
            if (j >= src.size()) {
                throw std::runtime_error("sql: unterminated string literal");
            }
            out.push_back({TokKind::StringLit, std::move(s)});
            i = j + 1;
            continue;
        }
        // Operators / punctuation: (, ), comma, =, !=, <, <=, >, >=
        if (c == '(' || c == ')' || c == ',') {
            out.push_back({TokKind::Op, std::string(1, c)});
            ++i;
            continue;
        }
        if (c == '!' && i + 1 < src.size() && src[i + 1] == '=') {
            out.push_back({TokKind::Op, "!="});
            i += 2;
            continue;
        }
        if (c == '<' || c == '>') {
            if (i + 1 < src.size() && src[i + 1] == '=') {
                out.push_back({TokKind::Op, std::string(1, c) + "="});
                i += 2;
            } else {
                out.push_back({TokKind::Op, std::string(1, c)});
                ++i;
            }
            continue;
        }
        if (c == '=') {
            out.push_back({TokKind::Op, "="});
            ++i;
            continue;
        }
        throw std::runtime_error(std::string("sql: unexpected character '") + c + "'");
    }
    out.push_back({TokKind::End, ""});
    return out;
}

// ---------------------------- Parser ------------------------------------

std::string upper(std::string s) {
    for (auto& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

struct Parser {
    const std::vector<Tok>& toks;
    std::size_t pos = 0;

    const Tok& peek() const { return toks[pos]; }
    const Tok& eat() { return toks[pos++]; }

    bool match_kw(const char* kw) {
        if (peek().kind == TokKind::Ident && upper(peek().text) == kw) {
            ++pos;
            return true;
        }
        return false;
    }
    void expect_kw(const char* kw) {
        if (!match_kw(kw)) {
            throw std::runtime_error(std::string("sql: expected keyword '") + kw + "', got '" +
                                     peek().text + "'");
        }
    }
    bool match_op(const std::string& s) {
        if (peek().kind == TokKind::Op && peek().text == s) {
            ++pos;
            return true;
        }
        return false;
    }
    void expect_op(const std::string& s) {
        if (!match_op(s)) {
            throw std::runtime_error("sql: expected '" + s + "', got '" + peek().text + "'");
        }
    }
    std::string expect_ident() {
        if (peek().kind != TokKind::Ident) {
            throw std::runtime_error("sql: expected identifier, got '" + peek().text + "'");
        }
        return eat().text;
    }

    AggKind parse_agg(const std::string& kw) {
        const std::string u = upper(kw);
        if (u == "SUM") {
            return AggKind::Sum;
        }
        if (u == "COUNT") {
            return AggKind::Count;
        }
        if (u == "MIN") {
            return AggKind::Min;
        }
        if (u == "MAX") {
            return AggKind::Max;
        }
        if (u == "AVG") {
            return AggKind::Avg;
        }
        if (u == "COUNT_DISTINCT") {
            return AggKind::CountDistinct;
        }
        return AggKind::None;
    }

    Projection parse_projection() {
        Projection p;
        if (peek().kind != TokKind::Ident) {
            throw std::runtime_error("sql: expected projection");
        }
        const std::string name = eat().text;
        const AggKind agg = parse_agg(name);
        if (agg != AggKind::None && peek().kind == TokKind::Op && peek().text == "(") {
            eat();
            p.col = expect_ident();
            expect_op(")");
            p.agg = agg;
            p.alias = upper(name) + "(" + p.col + ")";
        } else {
            p.col = name;
            p.alias = name;
        }
        return p;
    }

    CmpOp parse_cmp() {
        if (match_op("=")) {
            return CmpOp::Eq;
        }
        if (match_op("!=")) {
            return CmpOp::Ne;
        }
        if (match_op("<=")) {
            return CmpOp::Le;
        }
        if (match_op(">=")) {
            return CmpOp::Ge;
        }
        if (match_op("<")) {
            return CmpOp::Lt;
        }
        if (match_op(">")) {
            return CmpOp::Gt;
        }
        throw std::runtime_error("sql: expected comparison operator");
    }

    Query parse_query() {
        Query q;
        expect_kw("SELECT");
        q.projections.push_back(parse_projection());
        while (match_op(",")) {
            q.projections.push_back(parse_projection());
        }
        expect_kw("FROM");
        q.table = expect_ident();
        if (match_kw("WHERE")) {
            q.where.present = true;
            q.where.col = expect_ident();
            q.where.op = parse_cmp();
            if (peek().kind == TokKind::Number) {
                const std::string s = eat().text;
                if (s.find('.') != std::string::npos) {
                    q.where.value = std::stod(s);
                } else {
                    q.where.value = static_cast<int64_t>(std::stoll(s));
                }
            } else if (peek().kind == TokKind::StringLit) {
                q.where.value = eat().text;
            } else {
                throw std::runtime_error("sql: expected literal in WHERE");
            }
        }
        if (match_kw("GROUP")) {
            expect_kw("BY");
            q.group_by = expect_ident();
        }
        if (peek().kind != TokKind::End) {
            throw std::runtime_error("sql: unexpected trailing token '" + peek().text + "'");
        }
        return q;
    }
};

// ---------------------------- Executor ----------------------------------

double to_double(const io::CsvColumn& c, std::size_t i) {
    switch (c.type) {
    case io::ColType::Int32:
        return static_cast<double>(c.i32[i]);
    case io::ColType::Int64:
        return static_cast<double>(c.i64[i]);
    case io::ColType::Double:
        return c.f64[i];
    case io::ColType::String:
        throw std::runtime_error("sql: numeric expected, got string column '" + c.name + "'");
    }
    return 0.0;
}

bool row_matches(const io::CsvColumn& c,
                 std::size_t i,
                 CmpOp op,
                 const std::variant<int64_t, double, std::string>& v) {
    auto cmp = [&op](auto a, auto b) {
        switch (op) {
        case CmpOp::Eq:
            return a == b;
        case CmpOp::Ne:
            return a != b;
        case CmpOp::Lt:
            return a < b;
        case CmpOp::Le:
            return a <= b;
        case CmpOp::Gt:
            return a > b;
        case CmpOp::Ge:
            return a >= b;
        }
        return false;
    };
    if (c.type == io::ColType::String) {
        if (!std::holds_alternative<std::string>(v)) {
            throw std::runtime_error("sql: string column compared with non-string literal");
        }
        return cmp(c.str[i], std::get<std::string>(v));
    }
    const double a = to_double(c, i);
    double b;
    if (std::holds_alternative<int64_t>(v)) {
        b = static_cast<double>(std::get<int64_t>(v));
    } else if (std::holds_alternative<double>(v)) {
        b = std::get<double>(v);
    } else {
        throw std::runtime_error("sql: numeric column compared with string literal");
    }
    return cmp(a, b);
}

Cell column_cell(const io::CsvColumn& c, std::size_t i) {
    switch (c.type) {
    case io::ColType::Int32:
        return Cell{static_cast<int64_t>(c.i32[i])};
    case io::ColType::Int64:
        return Cell{static_cast<int64_t>(c.i64[i])};
    case io::ColType::Double:
        return Cell{c.f64[i]};
    case io::ColType::String:
        return Cell{c.str[i]};
    }
    return Cell{int64_t{0}};
}

std::string cell_key(const io::CsvColumn& c, std::size_t i) {
    switch (c.type) {
    case io::ColType::Int32:
        return std::to_string(c.i32[i]);
    case io::ColType::Int64:
        return std::to_string(c.i64[i]);
    case io::ColType::Double:
        return std::to_string(c.f64[i]);
    case io::ColType::String:
        return "s:" + c.str[i];
    }
    return "";
}

struct AggState {
    double sum = 0.0;
    int64_t count = 0;
    double min = 0.0;
    double max = 0.0;
    bool seen = false;
    bool source_is_double = false;
    std::set<std::string> distinct;
};

Cell finalize(AggKind kind, const AggState& s) {
    switch (kind) {
    case AggKind::Sum:
        if (s.source_is_double) {
            return Cell{s.sum};
        }
        return Cell{static_cast<int64_t>(s.sum)};
    case AggKind::Count:
        return Cell{s.count};
    case AggKind::Min:
        if (s.source_is_double) {
            return Cell{s.min};
        }
        return Cell{static_cast<int64_t>(s.min)};
    case AggKind::Max:
        if (s.source_is_double) {
            return Cell{s.max};
        }
        return Cell{static_cast<int64_t>(s.max)};
    case AggKind::Avg:
        return Cell{s.count == 0 ? 0.0 : s.sum / static_cast<double>(s.count)};
    case AggKind::CountDistinct:
        return Cell{static_cast<int64_t>(s.distinct.size())};
    case AggKind::None:
        return Cell{int64_t{0}};
    }
    return Cell{int64_t{0}};
}

void fold(AggKind kind, AggState& s, const io::CsvColumn& c, std::size_t i) {
    if (c.type == io::ColType::Double) {
        s.source_is_double = true;
    }
    const double v = (c.type == io::ColType::String) ? 0.0 : to_double(c, i);
    switch (kind) {
    case AggKind::Sum:
    case AggKind::Avg:
        s.sum += v;
        ++s.count;
        break;
    case AggKind::Count:
        ++s.count;
        break;
    case AggKind::Min:
        if (!s.seen) {
            s.min = v;
            s.seen = true;
        } else if (v < s.min) {
            s.min = v;
        }
        ++s.count;
        break;
    case AggKind::Max:
        if (!s.seen) {
            s.max = v;
            s.seen = true;
        } else if (v > s.max) {
            s.max = v;
        }
        ++s.count;
        break;
    case AggKind::CountDistinct:
        s.distinct.insert(cell_key(c, i));
        break;
    case AggKind::None:
        break;
    }
}

} // namespace

Query parse(const std::string& src) {
    auto toks = tokenize(src);
    Parser p{toks, 0};
    return p.parse_query();
}

ResultSet execute(const Query& q, const io::CsvTable& table) {
    // Validate projections.
    const io::CsvColumn* gb = nullptr;
    if (!q.group_by.empty()) {
        gb = table.find(q.group_by);
        if (!gb) {
            throw std::runtime_error("sql: GROUP BY column '" + q.group_by + "' not found");
        }
    }
    std::vector<const io::CsvColumn*> proj_cols(q.projections.size(), nullptr);
    for (std::size_t i = 0; i < q.projections.size(); ++i) {
        proj_cols[i] = table.find(q.projections[i].col);
        if (!proj_cols[i]) {
            throw std::runtime_error("sql: column '" + q.projections[i].col + "' not found");
        }
    }
    const io::CsvColumn* where_col = nullptr;
    if (q.where.present) {
        where_col = table.find(q.where.col);
        if (!where_col) {
            throw std::runtime_error("sql: WHERE column '" + q.where.col + "' not found");
        }
    }

    ResultSet rs;
    for (const auto& p : q.projections) {
        rs.column_names.push_back(p.alias);
    }

    const std::size_t N = table.row_count;

    if (q.group_by.empty()) {
        // Either pure projection (no agg) or single aggregate row.
        const bool any_agg = std::any_of(q.projections.begin(),
                                         q.projections.end(),
                                         [](const auto& p) { return p.agg != AggKind::None; });
        if (!any_agg) {
            for (std::size_t i = 0; i < N; ++i) {
                if (where_col && !row_matches(*where_col, i, q.where.op, q.where.value)) {
                    continue;
                }
                ResultRow row;
                for (std::size_t k = 0; k < q.projections.size(); ++k) {
                    row.cells.push_back(column_cell(*proj_cols[k], i));
                }
                rs.rows.push_back(std::move(row));
            }
            return rs;
        }
        std::vector<AggState> states(q.projections.size());
        for (std::size_t i = 0; i < N; ++i) {
            if (where_col && !row_matches(*where_col, i, q.where.op, q.where.value)) {
                continue;
            }
            for (std::size_t k = 0; k < q.projections.size(); ++k) {
                const auto& p = q.projections[k];
                fold(p.agg, states[k], *proj_cols[k], i);
            }
        }
        ResultRow row;
        for (std::size_t k = 0; k < q.projections.size(); ++k) {
            row.cells.push_back(finalize(q.projections[k].agg, states[k]));
        }
        rs.rows.push_back(std::move(row));
        return rs;
    }

    // GROUP BY path.
    std::map<std::string, std::vector<AggState>> groups;
    std::map<std::string, Cell> group_cell;
    for (std::size_t i = 0; i < N; ++i) {
        if (where_col && !row_matches(*where_col, i, q.where.op, q.where.value)) {
            continue;
        }
        const std::string k = cell_key(*gb, i);
        auto it = groups.find(k);
        if (it == groups.end()) {
            it = groups.emplace(k, std::vector<AggState>(q.projections.size())).first;
            group_cell.emplace(k, column_cell(*gb, i));
        }
        for (std::size_t kx = 0; kx < q.projections.size(); ++kx) {
            const auto& p = q.projections[kx];
            if (p.agg == AggKind::None) {
                // For a non-aggregate projection in a GROUP BY query, the
                // value is constant per group; we read it once and cache.
                continue;
            }
            fold(p.agg, it->second[kx], *proj_cols[kx], i);
        }
    }
    for (auto& [k, st] : groups) {
        ResultRow row;
        for (std::size_t kx = 0; kx < q.projections.size(); ++kx) {
            const auto& p = q.projections[kx];
            if (p.agg == AggKind::None) {
                // Non-agg projection: must equal group-by column or be a
                // functional dependency. For our subset, require col == gb.
                if (p.col != q.group_by) {
                    throw std::runtime_error(
                        "sql: non-aggregate projection '" + p.col +
                        "' must match GROUP BY column in this minimal frontend");
                }
                row.cells.push_back(group_cell.at(k));
            } else {
                row.cells.push_back(finalize(p.agg, st[kx]));
            }
        }
        rs.rows.push_back(std::move(row));
    }
    return rs;
}

} // namespace sql
} // namespace columnstore
