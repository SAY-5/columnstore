#include "simd/cpu_detect.h"

#include <cstdlib>
#include <cstring>

namespace columnstore {

namespace {

bool env_force_scalar() {
    const char* v = std::getenv("COLUMNSTORE_FORCE_SCALAR");
    if (v == nullptr) {
        return false;
    }
    return v[0] != '\0' && std::strcmp(v, "0") != 0;
}

} // namespace

SimdPath detect_simd_path() {
#if defined(COLUMNSTORE_FORCE_SCALAR)
    return SimdPath::Scalar;
#else
    if (env_force_scalar()) {
        return SimdPath::Scalar;
    }
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2")) {
        return SimdPath::Avx2;
    }
#endif
    return SimdPath::Scalar;
#endif
}

SimdPath active_simd_path() {
    static SimdPath cached = detect_simd_path();
    return cached;
}

const char* simd_path_name(SimdPath p) {
    switch (p) {
    case SimdPath::Scalar:
        return "scalar";
    case SimdPath::Avx2:
        return "avx2";
    }
    return "unknown";
}

} // namespace columnstore
