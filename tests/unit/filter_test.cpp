#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/batch.h"
#include "simd/cpu_detect.h"
#include "simd/kernels.h"

using namespace columnstore;

namespace {

[[maybe_unused]] std::vector<int32_t> gen(std::size_t n, uint64_t seed, int32_t lo, int32_t hi) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> d(lo, hi);
    std::vector<int32_t> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(d(rng));
    }
    return v;
}

} // namespace

TEST(FilterScalar, LessThanBasic) {
    std::vector<int32_t> data = {1, 5, 3, 7, 2, 8, 4, 6, 0, 9};
    std::vector<uint8_t> bm(Batch<int32_t>::bitmap_bytes(data.size()), 0);
    std::size_t passed = filter_int32_lt_scalar(data.data(), data.size(), 5, bm.data());
    EXPECT_EQ(passed, 5u);           // 1, 3, 2, 4, 0
    EXPECT_TRUE((bm[0] >> 0) & 1u);  // 1 < 5
    EXPECT_FALSE((bm[0] >> 1) & 1u); // 5 < 5
    EXPECT_TRUE((bm[0] >> 2) & 1u);  // 3 < 5
}

TEST(FilterScalar, GreaterThanBasic) {
    std::vector<int32_t> data = {1, 5, 3, 7, 2, 8, 4, 6, 0, 9};
    std::vector<uint8_t> bm(Batch<int32_t>::bitmap_bytes(data.size()), 0);
    std::size_t passed = filter_int32_gt_scalar(data.data(), data.size(), 5, bm.data());
    EXPECT_EQ(passed, 4u); // 7, 8, 6, 9
}

#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)

TEST(FilterAvx2VsScalar, BitExactRandom) {
    if (active_simd_path() != SimdPath::Avx2) {
        GTEST_SKIP() << "AVX2 not available at runtime";
    }
    std::mt19937_64 rng(0xDEADBEEFCAFEull);
    std::uniform_int_distribution<int32_t> dthr(-1'000'000, 1'000'000);
    std::uniform_int_distribution<std::size_t> dn(1, 65'536);

    for (int trial = 0; trial < 1000; ++trial) {
        const std::size_t n = dn(rng);
        const int32_t thr = dthr(rng);
        auto data = gen(n, rng(), -2'000'000, 2'000'000);

        std::vector<uint8_t> bm_avx2(Batch<int32_t>::bitmap_bytes(n), 0);
        std::vector<uint8_t> bm_sc(Batch<int32_t>::bitmap_bytes(n), 0);
        std::size_t c1 = filter_int32_lt_avx2(data.data(), n, thr, bm_avx2.data());
        std::size_t c2 = filter_int32_lt_scalar(data.data(), n, thr, bm_sc.data());
        ASSERT_EQ(c1, c2) << "trial=" << trial << " n=" << n << " thr=" << thr;
        // Bit-exact compare, but only the bits that matter (n bits).
        for (std::size_t i = 0; i < n; ++i) {
            const bool a = (bm_avx2[i / 8] >> (i % 8)) & 1u;
            const bool b = (bm_sc[i / 8] >> (i % 8)) & 1u;
            ASSERT_EQ(a, b) << "lt: trial=" << trial << " i=" << i;
        }

        std::fill(bm_avx2.begin(), bm_avx2.end(), 0);
        std::fill(bm_sc.begin(), bm_sc.end(), 0);
        c1 = filter_int32_gt_avx2(data.data(), n, thr, bm_avx2.data());
        c2 = filter_int32_gt_scalar(data.data(), n, thr, bm_sc.data());
        ASSERT_EQ(c1, c2);
        for (std::size_t i = 0; i < n; ++i) {
            const bool a = (bm_avx2[i / 8] >> (i % 8)) & 1u;
            const bool b = (bm_sc[i / 8] >> (i % 8)) & 1u;
            ASSERT_EQ(a, b) << "gt: trial=" << trial << " i=" << i;
        }
    }
}

TEST(FilterAvx2VsScalar, BoundaryThresholds) {
    if (active_simd_path() != SimdPath::Avx2) {
        GTEST_SKIP() << "AVX2 not available at runtime";
    }
    const int32_t lo = std::numeric_limits<int32_t>::min();
    const int32_t hi = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> data = {lo, -1, 0, 1, hi, lo + 1, hi - 1, 42};
    for (int32_t thr : {lo, -1, 0, 1, hi}) {
        std::vector<uint8_t> a(Batch<int32_t>::bitmap_bytes(data.size()), 0);
        std::vector<uint8_t> b(Batch<int32_t>::bitmap_bytes(data.size()), 0);
        filter_int32_lt_avx2(data.data(), data.size(), thr, a.data());
        filter_int32_lt_scalar(data.data(), data.size(), thr, b.data());
        EXPECT_EQ(a, b) << "thr=" << thr;
    }
}

TEST(FilterAvx2VsScalar, UnalignedAndShortTails) {
    if (active_simd_path() != SimdPath::Avx2) {
        GTEST_SKIP() << "AVX2 not available at runtime";
    }
    // n values that exercise both the 64-element block and the 8-element
    // tail and the < 8 scalar tail.
    for (std::size_t n : {1u, 7u, 8u, 9u, 15u, 16u, 63u, 64u, 65u, 71u, 4095u, 4096u, 4097u}) {
        auto data = gen(n, 12345u + n, -100, 100);
        std::vector<uint8_t> a(Batch<int32_t>::bitmap_bytes(n), 0);
        std::vector<uint8_t> b(Batch<int32_t>::bitmap_bytes(n), 0);
        filter_int32_lt_avx2(data.data(), n, 0, a.data());
        filter_int32_lt_scalar(data.data(), n, 0, b.data());
        for (std::size_t i = 0; i < n; ++i) {
            const bool av = (a[i / 8] >> (i % 8)) & 1u;
            const bool bv = (b[i / 8] >> (i % 8)) & 1u;
            ASSERT_EQ(av, bv) << "n=" << n << " i=" << i;
        }
    }
}

#endif
