# Scalar fallback

The engine ships scalar versions of every SIMD operator and selects
between them at process start via runtime CPU feature detection.

## Two gates

Compile-time:

```cmake
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2_FLAG)
if(COMPILER_SUPPORTS_AVX2_FLAG)
    set(COLUMNSTORE_HAS_AVX2 ON)
endif()
```

If the compiler supports `-mavx2`, the AVX2 sources are added to an
`OBJECT` library compiled with `-mavx2` only for those files. Everything
else stays baseline so the binary still loads on non-AVX2 hosts.

Runtime:

```cpp
SimdPath detect_simd_path() {
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2) && \
    (defined(__x86_64__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2")) return SimdPath::Avx2;
#endif
    return SimdPath::Scalar;
}
```

`__builtin_cpu_supports` is a libgcc/libcompiler-rt builtin that calls
into a static initializer reading `cpuid`. The result is cached for the
process lifetime. Without this gate, running an AVX2 binary on a
pre-Haswell CPU would crash with `SIGILL`.

The selected path is stored in a `static SimdPath` inside
`active_simd_path()`. Each operator reads it once per batch:

```cpp
const SimdPath path = active_simd_path();
if (path == SimdPath::Avx2) {
    passed = filter_int32_lt_avx2(...);
} else {
    passed = filter_int32_lt_scalar(...);
}
```

The cost is one branch on a value that never changes after process start,
trivially predicted by the CPU.

## Overrides

Two ways to force scalar:

- `cmake -DCOLUMNSTORE_FORCE_SCALAR=ON`: compile-time. Sets a `#define`
  that makes `detect_simd_path` unconditionally return scalar; the AVX2
  source files are still compiled but never called. CI uses this to
  validate the scalar test suite without playing CPUID games.
- `COLUMNSTORE_FORCE_SCALAR=1` env var: runtime. `detect_simd_path` reads
  this at the first call. Useful for triaging "is this a SIMD bug?"
  without rebuilding.

## ARM

The runtime always picks scalar on ARM. No hand-written NEON kernel is
shipped. Clang's auto-vectorizer will still produce NEON for the scalar
sum loop, so arm64 throughput is competitive on aggregate; the filter
kernel is harder to auto-vectorize because of the bitmap pack step.

If you need a hand-written NEON path, the layout is ready: add files like
`filter_int32_neon.cpp` next to the AVX2 ones, gate them on
`COLUMNSTORE_COMPILED_WITH_NEON` in CMake, add a `SimdPath::Neon`, and
extend `detect_simd_path` to pick it on ARM.

## Why both gates

The compile-time gate keeps the binary portable: a build on a non-AVX2
toolchain skips the AVX2 object entirely. The runtime gate keeps the
binary loadable on older CPUs: a build that does include the AVX2 object
still works on hosts where `__builtin_cpu_supports("avx2")` is false,
because the AVX2 code is never executed.

A "compile with `-mavx2` and let the loader crash on non-AVX2 hosts"
approach is simpler but unfit for a portfolio project that should run on
any x86_64 host, including pre-Haswell servers and CI runners that may
not advertise AVX2 (rare today but possible).
