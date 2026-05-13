#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace columnstore {
namespace bench {

inline std::size_t env_rows(std::size_t default_rows) {
    if (const char* env = std::getenv("COLUMNSTORE_BENCH_ROWS")) {
        return static_cast<std::size_t>(std::stoull(env));
    }
    return default_rows;
}

inline int env_iters(int default_iters) {
    if (const char* env = std::getenv("COLUMNSTORE_BENCH_ITERS")) {
        return std::atoi(env);
    }
    return default_iters;
}

inline std::vector<int32_t> synth_int32(std::size_t n, uint64_t seed) {
    std::vector<int32_t> v;
    v.reserve(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, 100'000);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(dist(rng));
    }
    return v;
}

inline uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

// Defeat optimizer dropping the result.
template <typename T> inline void sink(T& v) {
    asm volatile("" : "+r,m"(v) : : "memory");
}

inline void clobber() {
    asm volatile("" : : : "memory");
}

} // namespace bench
} // namespace columnstore
