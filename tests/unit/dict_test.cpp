#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/dict.h"
#include "exec/count_distinct.h"
#include "simd/cpu_detect.h"

using namespace columnstore;

namespace {

std::vector<int32_t> cardinality_8_rows(std::size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, 7);
    std::vector<int32_t> v;
    v.reserve(n);
    static const int32_t palette[8] = {-3, -2, -1, 0, 1, 2, 3, 99};
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(palette[dist(rng)]);
    }
    return v;
}

} // namespace

TEST(DictColumn, BuildLowCardinalityInt32) {
    auto raw = cardinality_8_rows(10'000, 0xA5A5);
    auto d = DictColumn<int32_t>::try_build(raw);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->cardinality(), 8u);
    EXPECT_EQ(d->row_count(), 10'000u);

    // Round-trip equal.
    auto back = d->materialize();
    ASSERT_EQ(back.size(), raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        ASSERT_EQ(back[i], raw[i]) << "row " << i;
    }
}

TEST(DictColumn, RejectsHighCardinality) {
    std::vector<int32_t> raw;
    raw.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        raw.push_back(i);
    }
    auto d = DictColumn<int32_t>::try_build(raw);
    EXPECT_FALSE(d.has_value());
}

TEST(DictColumn, BuildStringDict) {
    std::vector<std::string> raw = {"alpha", "beta", "alpha", "gamma", "beta", "alpha"};
    auto d = DictColumn<std::string>::try_build(raw);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->cardinality(), 3u);
    EXPECT_EQ(d->row_count(), 6u);
    auto back = d->materialize();
    ASSERT_EQ(back, raw);
}

TEST(CountDistinct, FastPathReturnsCardinalityImmediately) {
    auto raw = cardinality_8_rows(1'000'000, 0xC11);
    auto d = DictColumn<int32_t>::try_build(raw);
    ASSERT_TRUE(d.has_value());

    const auto t0 = std::chrono::steady_clock::now();
    const auto k_dict = CountDistinctInt32::run_dict(*d);
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_EQ(k_dict, 8u);
    const auto dict_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    const auto s0 = std::chrono::steady_clock::now();
    const auto k_scalar = CountDistinctInt32::run_scalar(raw.data(), raw.size());
    const auto s1 = std::chrono::steady_clock::now();
    EXPECT_EQ(k_scalar, 8u);
    const auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count();

    // The dict path is O(1) over the dictionary; the scalar path is O(N) over
    // 1M values into an unordered_set. The dict path should be vastly faster.
    // We require at least 1000x to leave plenty of slack for noisy hosts.
    std::cout << "[dict count_distinct] dict_ns=" << dict_ns << " scalar_ns=" << scalar_ns
              << " speedup=" << (scalar_ns ? double(scalar_ns) / double(dict_ns) : 0.0) << "x\n";
    EXPECT_LT(dict_ns * 1000, scalar_ns)
        << "dict path should be > 1000x faster than scalar on 1M rows / K=8";
}

TEST(CountDistinct, FilteredCountsOnlySelectedCodes) {
    auto raw = cardinality_8_rows(64, 0xC0DE);
    auto d = DictColumn<int32_t>::try_build(raw);
    ASSERT_TRUE(d.has_value());

    // Build a bitmap that keeps only rows whose code is even (codes 0,2,4,6).
    const std::size_t n = raw.size();
    std::vector<uint8_t> bm((n + 7) / 8, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if ((d->codes()[i] % 2) == 0) {
            bm[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
    }
    const auto k = CountDistinctInt32::run_dict_filtered(*d, bm.data(), n);
    EXPECT_EQ(k, 4u);
}

TEST(DictFilter, ScalarMatchesAvx2OnRandomCodes) {
    std::mt19937_64 rng(0xF111);
    std::uniform_int_distribution<int> dn(1, 8192);
    std::uniform_int_distribution<int> dc(0, 7);

    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = static_cast<std::size_t>(dn(rng));
        std::vector<uint8_t> codes(n);
        for (auto& c : codes) {
            c = static_cast<uint8_t>(dc(rng));
        }
        const uint8_t target = static_cast<uint8_t>(dc(rng));

        std::vector<uint8_t> bm_sc((n + 7) / 8, 0);
        std::vector<uint8_t> bm_av((n + 7) / 8, 0);
        const std::size_t sc = filter_dict_eq_scalar(codes.data(), n, target, bm_sc.data());
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (active_simd_path() == SimdPath::Avx2) {
            const std::size_t av = filter_dict_eq_avx2(codes.data(), n, target, bm_av.data());
            ASSERT_EQ(sc, av) << "trial=" << trial << " n=" << n << " target=" << int(target);
            for (std::size_t b = 0; b < (n + 7) / 8; ++b) {
                ASSERT_EQ(bm_sc[b], bm_av[b]) << "trial=" << trial << " byte=" << b << " n=" << n
                                              << " target=" << int(target);
            }
        }
#endif
        (void)sc;
    }
}
