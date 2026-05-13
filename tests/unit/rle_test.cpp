#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "rle/decoder.h"
#include "rle/encoder.h"

using namespace columnstore;

TEST(Rle, EncodeSimpleRuns) {
    std::vector<int32_t> in = {1, 1, 1, 2, 2, 3};
    auto enc = rle_encode_int32(in.data(), in.size());
    ASSERT_EQ(enc.values.size(), 3u);
    EXPECT_EQ(enc.values[0], 1);
    EXPECT_EQ(enc.lengths[0], 3u);
    EXPECT_EQ(enc.values[1], 2);
    EXPECT_EQ(enc.lengths[1], 2u);
    EXPECT_EQ(enc.values[2], 3);
    EXPECT_EQ(enc.lengths[2], 1u);
}

TEST(Rle, EncodeEmpty) {
    auto enc = rle_encode_int32(nullptr, 0);
    EXPECT_TRUE(enc.values.empty());
    EXPECT_TRUE(enc.lengths.empty());
}

TEST(Rle, EncodeAllDistinct) {
    std::vector<int32_t> in = {1, 2, 3, 4, 5};
    auto enc = rle_encode_int32(in.data(), in.size());
    EXPECT_EQ(enc.values.size(), 5u);
    for (auto l : enc.lengths) {
        EXPECT_EQ(l, 1u);
    }
}

TEST(Rle, DecoderRoundTripRandom) {
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> val_dist(0, 9);
    std::uniform_int_distribution<int> len_dist(1, 50);

    for (int trial = 0; trial < 50; ++trial) {
        std::vector<int32_t> original;
        const int runs = 200;
        for (int r = 0; r < runs; ++r) {
            const int v = val_dist(rng);
            const int len = len_dist(rng);
            for (int i = 0; i < len; ++i) {
                original.push_back(v);
            }
        }
        auto enc = rle_encode_int32(original.data(), original.size());

        RleDecoder dec(enc.values.data(), enc.lengths.data(), enc.values.size());
        std::vector<int32_t> recovered;
        RleBatch batch;
        while (true) {
            ASSERT_TRUE(dec.next_batch(batch));
            if (batch.size == 0) {
                break;
            }
            recovered.insert(recovered.end(), batch.buffer, batch.buffer + batch.size);
        }
        EXPECT_FALSE(dec.malformed());
        ASSERT_EQ(recovered.size(), original.size());
        EXPECT_EQ(recovered, original);
    }
}

TEST(Rle, DecoderBatchBoundary) {
    // Run length straddles a kBatchSize boundary.
    std::vector<int32_t> values = {42};
    std::vector<uint32_t> lengths = {static_cast<uint32_t>(kBatchSize + 17)};
    RleDecoder dec(values.data(), lengths.data(), 1);

    RleBatch b1;
    ASSERT_TRUE(dec.next_batch(b1));
    EXPECT_EQ(b1.size, kBatchSize);
    for (std::size_t i = 0; i < b1.size; ++i) {
        EXPECT_EQ(b1.buffer[i], 42);
    }
    RleBatch b2;
    ASSERT_TRUE(dec.next_batch(b2));
    EXPECT_EQ(b2.size, 17u);
    RleBatch b3;
    ASSERT_TRUE(dec.next_batch(b3));
    EXPECT_EQ(b3.size, 0u);
}

TEST(Rle, DecoderRejectsZeroLengthFirstRun) {
    std::vector<int32_t> values = {5};
    std::vector<uint32_t> lengths = {0};
    RleDecoder dec(values.data(), lengths.data(), 1);
    EXPECT_TRUE(dec.malformed());
    RleBatch b;
    EXPECT_FALSE(dec.next_batch(b));
}

TEST(Rle, DecoderDoesNotCrashOnEmpty) {
    RleDecoder dec(nullptr, nullptr, 0);
    RleBatch b;
    ASSERT_TRUE(dec.next_batch(b));
    EXPECT_EQ(b.size, 0u);
}
