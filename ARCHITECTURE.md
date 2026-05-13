# Architecture

## Vector-at-a-time execution

The pipeline is built from operators that exchange `Batch<int32_t>` values,
each up to `kBatchSize = 4096` rows. Each operator pulls a batch from its
child, processes it in place, and returns. There is no per-row virtual
dispatch in the hot path; the dispatch happens once per batch.

Why 4096:

- 4096 int32 == 16 KiB. Two such batches plus a bitmap fit in 32 KiB of L1d
  on Skylake-class cores.
- Large enough to amortize per-batch overhead (operator dispatch, bitmap
  allocation, function call) and to give the auto-vectorizer enough room.
- Small enough to keep the working set in cache across the
  filter -> aggregate hand-off, which is the operation that benefits most
  from data locality.

Why pull-based:

- Sinks decide how many rows to consume. An LIMIT operator would terminate
  early without waking the source.
- No queue between operators; each batch lives on the caller's stack.

Why not row-at-a-time:

- Modern CPUs execute several int operations per cycle and have wide SIMD
  units. Single-row code wastes that.
- Branch prediction breaks down at a predicate evaluated row-by-row through
  virtual calls. Batching turns the same predicate into one straight-line
  vector loop, eliminating both the misprediction penalty and the call
  overhead.

## AVX2 intrinsics: filter walkthrough

The hot loop in `src/simd/filter_int32_avx2.cpp`:

```cpp
const __m256i thr = _mm256_set1_epi32(threshold);
while (i + 64 <= n) {
    uint64_t bits = 0;
    for (int j = 0; j < 8; ++j) {
        const __m256i v =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + j * 8));
        const __m256i cmp = _mm256_cmpgt_epi32(thr, v);
        const int mask8 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        bits |= (static_cast<uint64_t>(mask8) & 0xFFull) << (j * 8);
    }
    std::memcpy(out_bitmap + i / 8, &bits, sizeof(bits));
    i += 64;
}
```

The shapes:

| step                              | input shape          | output shape    |
| --------------------------------- | -------------------- | --------------- |
| `_mm256_set1_epi32(thr)`          | scalar               | 8 lanes x int32 |
| `_mm256_loadu_si256(data+i)`      | 8 int32 from memory  | 8 lanes x int32 |
| `_mm256_cmpgt_epi32(thr, v)`      | thr, v               | 0xFFFFFFFF or 0 per lane |
| `_mm256_castsi256_ps(cmp)`        | reinterpret bits     | 8 lanes x float |
| `_mm256_movemask_ps(...)`         | 8 float lanes        | 8-bit mask      |

Packing 8 of those 8-bit masks into a `uint64_t` gives 64 rows of output
bitmap. One `memcpy` per 64 rows; the load and the compare are on the same
data the prefetcher already pulled in.

Why `movemask_ps` and not `movemask_epi8`:

- `_mm256_movemask_epi8` returns 32 bits, one per byte. For an int32 lane,
  that emits the same bit four times (because the 0xFFFFFFFF/0 result fills
  all four bytes of the lane). Picking out a clean 8-bit mask would need a
  per-fourth-bit pdep.
- `_mm256_movemask_ps` returns 8 bits, one per 32-bit lane. Exactly the
  shape we want.

Tail handling:

- After the 64-row main loop, we run an 8-row tail loop that writes one
  bitmap byte per pass.
- After that, a final scalar loop covers the < 8 leftover rows.

## RLE encoding format

Runs are stored as two parallel arrays:

- `values[k]`: the value of run k (int32)
- `lengths[k]`: the run length of run k (uint32, must be > 0)

Adjacent equal values are coalesced at encode time. A pathological run
longer than `UINT32_MAX` splits across two records; in practice no run
exceeds the source buffer size.

When RLE helps:

- Sorted or near-sorted low-cardinality columns. A column of 100M values
  drawn from 100 distinct levels with average run length 8 compresses to
  ~12.5 M (value, length) pairs, a 4 x reduction in bytes scanned.
- Dictionary-encoded columns (deferred).

When RLE hurts:

- Random columns. Average run length is ~1, the (value, length) pair is
  twice the original size, and the per-run branch in the decoder limits
  the throughput vs. a flat scan.

The decoder is streaming: each `next_batch` call emits up to `kBatchSize`
values into a stack-allocated buffer. The source operator forwards that
buffer downstream. No materialization happens unless the caller asks for
it (`Column<T>::materialize`).

## Scalar fallback

CPU dispatch is one read at process start, cached for the lifetime of the
process.

```cpp
SimdPath detect_simd_path() {
#if defined(COLUMNSTORE_FORCE_SCALAR)
    return SimdPath::Scalar;
#else
    if (env_force_scalar()) return SimdPath::Scalar;
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2) && \
    (defined(__x86_64__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2")) return SimdPath::Avx2;
#endif
    return SimdPath::Scalar;
#endif
}
```

Two gates:

- Compile time: `COLUMNSTORE_COMPILED_WITH_AVX2` is set by CMake only when
  the compiler supports `-mavx2`. AVX2 source files live in their own
  `OBJECT` library compiled with `-mavx2` so the rest of the program stays
  baseline.
- Run time: `__builtin_cpu_supports("avx2")` confirms the running CPU has
  AVX2. If not, we fall back to scalar; the AVX2 object is linked but
  unreachable.

Override: `COLUMNSTORE_FORCE_SCALAR=1` (env) forces scalar at runtime.
`-DCOLUMNSTORE_FORCE_SCALAR=ON` (cmake) forces it at compile time and
removes the AVX2 object from the binary.

ARM: no NEON kernel is shipped. The runtime always picks scalar. Clang's
auto-vectorizer still produces NEON for the scalar `sum` loop, which is
why arm64 throughput on aggregate is close to AVX2-saturated.

## Bench methodology

Each bench:

- Allocates the input vector once.
- Runs one warmup iteration.
- Records 5 iterations, reports min (best) wall-clock and p50/p95/p99 over
  iteration latency.
- Uses `std::chrono::steady_clock`, ns precision.
- Defeats the optimizer with an inline-asm `sink` on the result.

CI bench-smoke runs at `rows=1M`, `iters=3`, and asserts a 100M v/s floor on
the scan pipeline. Local-host runs use the default `rows=100M`. Cache
state: the second iteration onward runs on a fully cached input (best case
for AVX2 throughput); the first iteration warms it. We report best-of-5,
which means the reported number is the cache-warm best.
