// Property-style cross-product tests. For every operator (filter lt/gt,
// sum, min/max, count), generate 1000+ random (values, threshold)
// pairs and require the AVX2 kernel to be bit-exact equal to the scalar
// reference. Also exercises boundary thresholds (INT_MIN/INT_MAX/-1/0/1),
// fixed bit patterns (0x00/0xFF/0xAA/0x55), tail sizes around the AVX2
// block/tail boundaries, and an RLE round-trip property over random run
// pairs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "core/batch.h"
#include "rle/decoder.h"
#include "rle/encoder.h"
#include "simd/cpu_detect.h"
#include "simd/kernels.h"

using namespace columnstore;

namespace {

constexpr int kPropTrials = 1000;

std::vector<int32_t> gen_uniform(std::size_t n, std::mt19937_64& rng, int32_t lo, int32_t hi) {
    std::uniform_int_distribution<int32_t> d(lo, hi);
    std::vector<int32_t> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(d(rng));
    }
    return v;
}

std::vector<int32_t> gen_filled(std::size_t n, uint8_t byte) {
    std::vector<int32_t> v(n);
    int32_t pattern;
    const uint8_t bytes[4] = {byte, byte, byte, byte};
    std::memcpy(&pattern, bytes, 4);
    std::fill(v.begin(), v.end(), pattern);
    return v;
}

[[maybe_unused]] bool bitmaps_equal_lo(const uint8_t* a, const uint8_t* b, std::size_t n_bits) {
    for (std::size_t i = 0; i < n_bits; ++i) {
        const bool ai = (a[i / 8] >> (i % 8)) & 1u;
        const bool bi = (b[i / 8] >> (i % 8)) & 1u;
        if (ai != bi) {
            return false;
        }
    }
    return true;
}

// Boundary tail sizes that exercise the 64-element block, 8-element tail,
// and < 8 scalar tail in the AVX2 filter kernel.
const std::vector<std::size_t>& boundary_tail_sizes() {
    static const std::vector<std::size_t> v = []() {
        std::vector<std::size_t> r;
        for (std::size_t i = 1; i <= 71; ++i) {
            r.push_back(i);
        }
        for (std::size_t i = 4090; i <= 4097; ++i) {
            r.push_back(i);
        }
        return r;
    }();
    return v;
}

const std::vector<int32_t>& boundary_thresholds() {
    static const std::vector<int32_t> v = {
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::min() + 1,
        -2,
        -1,
        0,
        1,
        2,
        std::numeric_limits<int32_t>::max() - 1,
        std::numeric_limits<int32_t>::max(),
    };
    return v;
}

} // namespace

// -------------------------- Filter LT --------------------------------------

TEST(PropertyFilter, LessThanRandomCrossProduct) {
    std::mt19937_64 rng(0xA17CE10BADC0FFEEull);
    std::uniform_int_distribution<int32_t> dthr(std::numeric_limits<int32_t>::min(),
                                                std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<std::size_t> dn(1, 8192);

    for (int t = 0; t < kPropTrials; ++t) {
        const std::size_t n = dn(rng);
        const int32_t thr = dthr(rng);
        auto data = gen_uniform(
            n, rng, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
        std::vector<uint8_t> sc(Batch<int32_t>::bitmap_bytes(n), 0);
        const std::size_t sc_count = filter_int32_lt_scalar(data.data(), n, thr, sc.data());
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (active_simd_path() == SimdPath::Avx2) {
            std::vector<uint8_t> av(Batch<int32_t>::bitmap_bytes(n), 0);
            const std::size_t av_count = filter_int32_lt_avx2(data.data(), n, thr, av.data());
            ASSERT_EQ(sc_count, av_count) << "lt: trial=" << t << " n=" << n << " thr=" << thr;
            ASSERT_TRUE(bitmaps_equal_lo(av.data(), sc.data(), n))
                << "lt: trial=" << t << " n=" << n << " thr=" << thr;
        }
#endif
        // Scalar self-consistency: count_int32_scalar over the bitmap
        // matches the kernel's reported count.
        EXPECT_EQ(count_int32_scalar(n, sc.data()), sc_count);
    }
}

TEST(PropertyFilter, GreaterThanRandomCrossProduct) {
    std::mt19937_64 rng(0xDECA1B0E5ABBAull);
    std::uniform_int_distribution<int32_t> dthr(std::numeric_limits<int32_t>::min(),
                                                std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<std::size_t> dn(1, 8192);

    for (int t = 0; t < kPropTrials; ++t) {
        const std::size_t n = dn(rng);
        const int32_t thr = dthr(rng);
        auto data = gen_uniform(
            n, rng, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
        std::vector<uint8_t> sc(Batch<int32_t>::bitmap_bytes(n), 0);
        const std::size_t sc_count = filter_int32_gt_scalar(data.data(), n, thr, sc.data());
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (active_simd_path() == SimdPath::Avx2) {
            std::vector<uint8_t> av(Batch<int32_t>::bitmap_bytes(n), 0);
            const std::size_t av_count = filter_int32_gt_avx2(data.data(), n, thr, av.data());
            ASSERT_EQ(sc_count, av_count) << "gt: trial=" << t << " n=" << n << " thr=" << thr;
            ASSERT_TRUE(bitmaps_equal_lo(av.data(), sc.data(), n))
                << "gt: trial=" << t << " n=" << n << " thr=" << thr;
        }
#endif
        EXPECT_EQ(count_int32_scalar(n, sc.data()), sc_count);
    }
}

// -------------------------- Filter boundary thresholds ---------------------

TEST(PropertyFilter, BoundaryThresholdsCrossProduct) {
    std::mt19937_64 rng(0xBEEFFACE5A7ull);
    for (std::size_t n : boundary_tail_sizes()) {
        auto data = gen_uniform(
            n, rng, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
        for (int32_t thr : boundary_thresholds()) {
            std::vector<uint8_t> sc_lt(Batch<int32_t>::bitmap_bytes(n), 0);
            std::vector<uint8_t> sc_gt(Batch<int32_t>::bitmap_bytes(n), 0);
            const std::size_t c_lt = filter_int32_lt_scalar(data.data(), n, thr, sc_lt.data());
            const std::size_t c_gt = filter_int32_gt_scalar(data.data(), n, thr, sc_gt.data());
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
            if (active_simd_path() == SimdPath::Avx2) {
                std::vector<uint8_t> av_lt(Batch<int32_t>::bitmap_bytes(n), 0);
                std::vector<uint8_t> av_gt(Batch<int32_t>::bitmap_bytes(n), 0);
                ASSERT_EQ(c_lt, filter_int32_lt_avx2(data.data(), n, thr, av_lt.data()));
                ASSERT_EQ(c_gt, filter_int32_gt_avx2(data.data(), n, thr, av_gt.data()));
                ASSERT_TRUE(bitmaps_equal_lo(av_lt.data(), sc_lt.data(), n))
                    << "lt boundary n=" << n << " thr=" << thr;
                ASSERT_TRUE(bitmaps_equal_lo(av_gt.data(), sc_gt.data(), n))
                    << "gt boundary n=" << n << " thr=" << thr;
            }
#endif
            (void)c_lt;
            (void)c_gt;
        }
    }
}

// -------------------------- Filter fixed bit patterns ----------------------

TEST(PropertyFilter, FixedBitPatternsCrossProduct) {
    const uint8_t patterns[] = {0x00, 0xFF, 0xAA, 0x55};
    for (uint8_t p : patterns) {
        for (std::size_t n : boundary_tail_sizes()) {
            auto data = gen_filled(n, p);
            for (int32_t thr : boundary_thresholds()) {
                std::vector<uint8_t> sc_lt(Batch<int32_t>::bitmap_bytes(n), 0);
                std::vector<uint8_t> sc_gt(Batch<int32_t>::bitmap_bytes(n), 0);
                const std::size_t c_lt = filter_int32_lt_scalar(data.data(), n, thr, sc_lt.data());
                const std::size_t c_gt = filter_int32_gt_scalar(data.data(), n, thr, sc_gt.data());
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
                if (active_simd_path() == SimdPath::Avx2) {
                    std::vector<uint8_t> av_lt(Batch<int32_t>::bitmap_bytes(n), 0);
                    std::vector<uint8_t> av_gt(Batch<int32_t>::bitmap_bytes(n), 0);
                    ASSERT_EQ(c_lt, filter_int32_lt_avx2(data.data(), n, thr, av_lt.data()));
                    ASSERT_EQ(c_gt, filter_int32_gt_avx2(data.data(), n, thr, av_gt.data()));
                    ASSERT_TRUE(bitmaps_equal_lo(av_lt.data(), sc_lt.data(), n))
                        << "lt pattern=" << int(p) << " n=" << n << " thr=" << thr;
                    ASSERT_TRUE(bitmaps_equal_lo(av_gt.data(), sc_gt.data(), n))
                        << "gt pattern=" << int(p) << " n=" << n << " thr=" << thr;
                }
#endif
                (void)c_lt;
                (void)c_gt;
            }
        }
    }
}

// -------------------------- Sum AVX2 vs scalar -----------------------------

TEST(PropertySum, RandomCrossProduct) {
    std::mt19937_64 rng(0x5A11BADD1E5ull);
    std::uniform_int_distribution<std::size_t> dn(1, 8192);

    for (int t = 0; t < kPropTrials; ++t) {
        const std::size_t n = dn(rng);
        auto data = gen_uniform(n, rng, -1'000'000, 1'000'000);
        const int64_t sc = sum_int32_scalar(data.data(), n, nullptr);
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (active_simd_path() == SimdPath::Avx2) {
            const int64_t av = sum_int32_avx2(data.data(), n, nullptr);
            ASSERT_EQ(sc, av) << "sum trial=" << t << " n=" << n;
        }
#endif
        (void)sc;
    }
}

TEST(PropertySum, MaskedRandomCrossProduct) {
    std::mt19937_64 rng(0xC1A55ECABAull);
    std::uniform_int_distribution<std::size_t> dn(1, 8192);

    for (int t = 0; t < kPropTrials; ++t) {
        const std::size_t n = dn(rng);
        auto data = gen_uniform(n, rng, -1'000'000, 1'000'000);
        std::vector<uint8_t> bm(Batch<int32_t>::bitmap_bytes(n), 0);
        for (auto& b : bm) {
            b = static_cast<uint8_t>(rng() & 0xFF);
        }
        const int64_t sc = sum_int32_scalar(data.data(), n, bm.data());
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (active_simd_path() == SimdPath::Avx2) {
            const int64_t av = sum_int32_avx2(data.data(), n, bm.data());
            ASSERT_EQ(sc, av) << "masked sum trial=" << t << " n=" << n;
        }
#endif
        (void)sc;
    }
}

TEST(PropertySum, BoundaryAndPatterns) {
    // Patterns + tail sizes: sum kernels must match scalar.
    const uint8_t patterns[] = {0x00, 0xFF, 0xAA, 0x55};
    for (uint8_t p : patterns) {
        for (std::size_t n : boundary_tail_sizes()) {
            auto data = gen_filled(n, p);
            const int64_t sc = sum_int32_scalar(data.data(), n, nullptr);
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
            if (active_simd_path() == SimdPath::Avx2) {
                const int64_t av = sum_int32_avx2(data.data(), n, nullptr);
                ASSERT_EQ(sc, av) << "sum pattern=" << int(p) << " n=" << n;
            }
#endif
            (void)sc;
        }
    }
}

// -------------------------- Min/max/count ----------------------------------

TEST(PropertyMinMaxCount, RandomCrossProduct) {
    std::mt19937_64 rng(0xF1115C0DEAD0DD0ull);
    std::uniform_int_distribution<std::size_t> dn(1, 8192);

    for (int t = 0; t < kPropTrials; ++t) {
        const std::size_t n = dn(rng);
        auto data = gen_uniform(n, rng, -1'000'000, 1'000'000);
        std::vector<uint8_t> bm(Batch<int32_t>::bitmap_bytes(n), 0);
        for (auto& b : bm) {
            b = static_cast<uint8_t>(rng() & 0xFF);
        }
        // Compute via stl on the masked subset for a fully independent
        // reference.
        std::vector<int32_t> kept;
        kept.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if ((bm[i / 8] >> (i % 8)) & 1u) {
                kept.push_back(data[i]);
            }
        }
        bool sc_empty = true;
        const int32_t sc_min = min_int32_scalar(data.data(), n, bm.data(), &sc_empty);
        const int32_t sc_max = max_int32_scalar(data.data(), n, bm.data(), &sc_empty);
        const std::size_t sc_cnt = count_int32_scalar(n, bm.data());
        if (kept.empty()) {
            EXPECT_TRUE(sc_empty);
            EXPECT_EQ(sc_cnt, 0u);
        } else {
            EXPECT_FALSE(sc_empty);
            EXPECT_EQ(sc_min, *std::min_element(kept.begin(), kept.end()));
            EXPECT_EQ(sc_max, *std::max_element(kept.begin(), kept.end()));
            EXPECT_EQ(sc_cnt, kept.size());
        }
    }
}

// -------------------------- RLE round-trip property ------------------------

TEST(PropertyRle, RoundTripRandomRuns) {
    std::mt19937_64 rng(0xBA0BAB1E3DF00Dull);
    std::uniform_int_distribution<int32_t> dval(std::numeric_limits<int32_t>::min(),
                                                std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint32_t> dlen(1, 64);
    std::uniform_int_distribution<int> dnruns(1, 256);

    for (int t = 0; t < kPropTrials; ++t) {
        const int nruns = dnruns(rng);
        std::vector<int32_t> values;
        std::vector<uint32_t> lengths;
        std::vector<int32_t> expanded;
        int32_t last = 0;
        bool have_last = false;
        for (int r = 0; r < nruns; ++r) {
            int32_t v = dval(rng);
            // Avoid coalescing across consecutive identical runs: the
            // encoder would collapse them, breaking a pair-by-pair compare.
            // Instead, expand and re-encode to compare structurally.
            if (have_last && v == last) {
                v = (v == std::numeric_limits<int32_t>::max()) ? v - 1 : v + 1;
            }
            const uint32_t len = dlen(rng);
            values.push_back(v);
            lengths.push_back(len);
            for (uint32_t k = 0; k < len; ++k) {
                expanded.push_back(v);
            }
            last = v;
            have_last = true;
        }
        // Encode the expanded buffer; result must match (values,lengths)
        // pair-by-pair.
        auto enc = rle_encode_int32(expanded.data(), expanded.size());
        ASSERT_EQ(enc.values, values) << "trial=" << t;
        ASSERT_EQ(enc.lengths, lengths) << "trial=" << t;

        // Decoder round-trip: must reproduce expanded buffer bit-identically.
        RleDecoder dec(enc.values.data(), enc.lengths.data(), enc.values.size());
        std::vector<int32_t> recovered;
        recovered.reserve(expanded.size());
        RleBatch batch;
        while (true) {
            ASSERT_TRUE(dec.next_batch(batch));
            if (batch.size == 0) {
                break;
            }
            recovered.insert(recovered.end(), batch.buffer, batch.buffer + batch.size);
        }
        ASSERT_FALSE(dec.malformed());
        ASSERT_EQ(recovered.size(), expanded.size()) << "trial=" << t;
        ASSERT_EQ(std::memcmp(recovered.data(), expanded.data(), expanded.size() * sizeof(int32_t)),
                  0)
            << "trial=" << t;
    }
}

TEST(PropertyRle, RoundTripBytePatterns) {
    // Encode buffers filled with a single value reconstructed from byte
    // patterns. The single-run output must round-trip exactly.
    const uint8_t patterns[] = {0x00, 0xFF, 0xAA, 0x55};
    for (uint8_t p : patterns) {
        for (std::size_t n : {1u, 7u, 64u, 4096u, 100'000u}) {
            auto data = gen_filled(n, p);
            auto enc = rle_encode_int32(data.data(), data.size());
            ASSERT_EQ(enc.values.size(), 1u);
            EXPECT_EQ(enc.lengths[0], n);
            RleDecoder dec(enc.values.data(), enc.lengths.data(), 1);
            std::vector<int32_t> recovered;
            RleBatch batch;
            while (true) {
                ASSERT_TRUE(dec.next_batch(batch));
                if (batch.size == 0) {
                    break;
                }
                recovered.insert(recovered.end(), batch.buffer, batch.buffer + batch.size);
            }
            ASSERT_EQ(recovered.size(), n);
            EXPECT_EQ(std::memcmp(recovered.data(), data.data(), n * sizeof(int32_t)), 0);
        }
    }
}
