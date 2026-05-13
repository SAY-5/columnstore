#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "core/batch.h"
#include "simd/cpu_detect.h"
#include "simd/kernels.h"

using namespace columnstore;

namespace {

[[maybe_unused]] std::vector<int32_t> gen(std::size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> d(-1'000'000, 1'000'000);
    std::vector<int32_t> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(d(rng));
    }
    return v;
}

[[maybe_unused]] std::vector<uint8_t> random_bitmap(std::size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> bm(Batch<int32_t>::bitmap_bytes(n), 0);
    for (auto& b : bm) {
        b = static_cast<uint8_t>(rng() & 0xFF);
    }
    return bm;
}

} // namespace

TEST(SumScalar, NoBitmap) {
    std::vector<int32_t> v = {1, 2, 3, 4, 5};
    EXPECT_EQ(sum_int32_scalar(v.data(), v.size(), nullptr), 15);
}

TEST(SumScalar, WithBitmap) {
    std::vector<int32_t> v = {10, 20, 30, 40, 50, 60, 70, 80};
    uint8_t bm = 0b10101010; // keep indices 1, 3, 5, 7
    EXPECT_EQ(sum_int32_scalar(v.data(), v.size(), &bm), 20 + 40 + 60 + 80);
}

TEST(MinMaxScalar, Basic) {
    std::vector<int32_t> v = {7, -3, 12, 4, 0};
    bool empty = true;
    EXPECT_EQ(min_int32_scalar(v.data(), v.size(), nullptr, &empty), -3);
    EXPECT_FALSE(empty);
    EXPECT_EQ(max_int32_scalar(v.data(), v.size(), nullptr, &empty), 12);
    EXPECT_FALSE(empty);
}

TEST(CountScalar, NoBitmapAndBitmap) {
    std::vector<int32_t> v(33, 1);
    EXPECT_EQ(count_int32_scalar(v.size(), nullptr), 33u);
    std::vector<uint8_t> bm = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    EXPECT_EQ(count_int32_scalar(33u, bm.data()), 33u);
}

#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)

TEST(SumAvx2VsScalar, BitExactNoBitmap) {
    if (active_simd_path() != SimdPath::Avx2) {
        GTEST_SKIP() << "AVX2 not available at runtime";
    }
    std::mt19937_64 rng(0x123456u);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + (rng() % 65'536);
        auto v = gen(n, rng());
        const int64_t a = sum_int32_avx2(v.data(), n, nullptr);
        const int64_t b = sum_int32_scalar(v.data(), n, nullptr);
        ASSERT_EQ(a, b) << "trial=" << trial << " n=" << n;
    }
}

TEST(SumAvx2VsScalar, BitExactWithBitmap) {
    if (active_simd_path() != SimdPath::Avx2) {
        GTEST_SKIP() << "AVX2 not available at runtime";
    }
    std::mt19937_64 rng(0x654321u);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + (rng() % 65'536);
        auto v = gen(n, rng());
        auto bm = random_bitmap(n, rng());
        const int64_t a = sum_int32_avx2(v.data(), n, bm.data());
        const int64_t b = sum_int32_scalar(v.data(), n, bm.data());
        ASSERT_EQ(a, b) << "trial=" << trial << " n=" << n;
    }
}

TEST(SumAvx2VsScalar, EdgeSizes) {
    if (active_simd_path() != SimdPath::Avx2) {
        GTEST_SKIP() << "AVX2 not available at runtime";
    }
    for (std::size_t n : {1u, 7u, 8u, 9u, 64u, 255u, 256u, 257u, 4096u}) {
        auto v = gen(n, 0xABCDu + n);
        const int64_t a = sum_int32_avx2(v.data(), n, nullptr);
        const int64_t b = sum_int32_scalar(v.data(), n, nullptr);
        ASSERT_EQ(a, b) << "n=" << n;
    }
}

#endif
