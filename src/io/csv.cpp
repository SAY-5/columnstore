#include "io/csv.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace columnstore {
namespace io {

namespace {

std::string trim(const std::string& s) {
    std::size_t a = 0;
    std::size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) {
        ++a;
    }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) {
        --b;
    }
    return s.substr(a, b - a);
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(trim(cur));
    return out;
}

ColType parse_type(const std::string& s) {
    if (s == "int32") {
        return ColType::Int32;
    }
    if (s == "int64") {
        return ColType::Int64;
    }
    if (s == "double" || s == "float64") {
        return ColType::Double;
    }
    if (s == "string") {
        return ColType::String;
    }
    throw std::runtime_error("csv: unknown column type '" + s + "'");
}

} // namespace

const char* col_type_name(ColType t) {
    switch (t) {
    case ColType::Int32:
        return "int32";
    case ColType::Int64:
        return "int64";
    case ColType::Double:
        return "double";
    case ColType::String:
        return "string";
    }
    return "?";
}

std::size_t CsvColumn::row_count() const {
    switch (type) {
    case ColType::Int32:
        return i32.size();
    case ColType::Int64:
        return i64.size();
    case ColType::Double:
        return f64.size();
    case ColType::String:
        return str.size();
    }
    return 0;
}

CsvTable read_csv_string(const std::string& content) {
    std::istringstream is(content);
    std::string line;

    // First non-empty line: # types: t1, t2, ...
    std::vector<ColType> types;
    std::vector<std::string> names;
    bool have_types = false;
    bool have_header = false;

    CsvTable table;

    while (std::getline(is, line)) {
        const std::string t = trim(line);
        if (t.empty()) {
            continue;
        }

        if (!have_types) {
            if (t.rfind("# types:", 0) != 0 && t.rfind("#types:", 0) != 0) {
                throw std::runtime_error("csv: first non-empty line must be '# types: ...'");
            }
            const std::size_t colon = t.find(':');
            const std::string tail = t.substr(colon + 1);
            for (const auto& s : split_commas(tail)) {
                types.push_back(parse_type(s));
            }
            have_types = true;
            continue;
        }

        if (!have_header) {
            for (const auto& s : split_commas(t)) {
                names.push_back(s);
            }
            if (names.size() != types.size()) {
                throw std::runtime_error("csv: header column count != types count");
            }
            table.cols.resize(names.size());
            for (std::size_t i = 0; i < names.size(); ++i) {
                table.cols[i].name = names[i];
                table.cols[i].type = types[i];
            }
            have_header = true;
            continue;
        }

        // Data row.
        auto vals = split_commas(t);
        if (vals.size() != types.size()) {
            throw std::runtime_error("csv: data row column count mismatch");
        }
        for (std::size_t i = 0; i < vals.size(); ++i) {
            auto& c = table.cols[i];
            const auto& v = vals[i];
            switch (c.type) {
            case ColType::Int32:
                c.i32.push_back(static_cast<int32_t>(std::stol(v)));
                break;
            case ColType::Int64:
                c.i64.push_back(static_cast<int64_t>(std::stoll(v)));
                break;
            case ColType::Double:
                c.f64.push_back(std::stod(v));
                break;
            case ColType::String:
                c.str.push_back(v);
                break;
            }
        }
        ++table.row_count;
    }

    if (!have_header) {
        throw std::runtime_error("csv: missing header row");
    }
    return table;
}

CsvTable read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) {
        throw std::runtime_error("csv: cannot open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return read_csv_string(ss.str());
}

} // namespace io
} // namespace columnstore
