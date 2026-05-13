#include <gtest/gtest.h>

#include <vector>

#include "core/column.h"
#include "core/schema.h"
#include "rle/encoder.h"

using namespace columnstore;

TEST(Column, RawConstruction) {
    Column<int32_t> c = Column<int32_t>::from_raw({1, 2, 3, 4, 5});
    EXPECT_FALSE(c.is_rle());
    EXPECT_EQ(c.row_count(), 5u);
    EXPECT_EQ(c.raw_data()[0], 1);
    EXPECT_EQ(c.raw_data()[4], 5);
}

TEST(Column, RleConstructionAndMaterialize) {
    Column<int32_t> c = Column<int32_t>::from_rle({7, 9}, {3, 2});
    EXPECT_TRUE(c.is_rle());
    EXPECT_EQ(c.row_count(), 5u);
    auto flat = c.materialize();
    ASSERT_EQ(flat.size(), 5u);
    EXPECT_EQ(flat[0], 7);
    EXPECT_EQ(flat[1], 7);
    EXPECT_EQ(flat[2], 7);
    EXPECT_EQ(flat[3], 9);
    EXPECT_EQ(flat[4], 9);
}

TEST(Column, EmptyMaterialize) {
    Column<int32_t> c;
    EXPECT_EQ(c.row_count(), 0u);
    EXPECT_TRUE(c.materialize().empty());
}

TEST(StringColumn, AppendAndRead) {
    StringColumn s;
    s.append("hello");
    s.append("");
    s.append("world!");
    ASSERT_EQ(s.row_count(), 3u);
    EXPECT_EQ(s.at(0), "hello");
    EXPECT_EQ(s.at(1), "");
    EXPECT_EQ(s.at(2), "world!");
}

TEST(Schema, AddAndLookup) {
    Schema sc;
    sc.add_column("id", DataType::Int64);
    sc.add_column("value", DataType::Int32);
    sc.add_column("label", DataType::String);
    EXPECT_EQ(sc.size(), 3u);
    EXPECT_EQ(sc.index_of("value"), 1);
    EXPECT_EQ(sc.index_of("missing"), -1);
    EXPECT_EQ(sc.at(2).type, DataType::String);
}
