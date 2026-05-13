#pragma once

namespace columnstore {

enum class SimdPath {
    Scalar = 0,
    Avx2 = 1,
};

// Runtime feature detection. Reads ColumnStore env vars and __builtin_cpu_supports.
// Honors COLUMNSTORE_FORCE_SCALAR=1 at runtime (in addition to the
// compile-time define).
SimdPath detect_simd_path();

// Cached path selected at process start.
SimdPath active_simd_path();

const char* simd_path_name(SimdPath p);

} // namespace columnstore
