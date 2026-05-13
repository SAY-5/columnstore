# columnstore

A C++17 in-memory column-store query engine with SIMD-accelerated filter and
aggregate operators (AVX2 intrinsics on x86_64), run-length-encoded
compression, and a vector-at-a-time execution model that processes columns in
batches of 4096 values. CPU-feature detection picks the AVX2 or scalar path
at runtime, so the same binary works on hosts without AVX2.

## What this studies

- Vector-at-a-time execution. Operators pull one fixed-size batch from their
  child, instead of one row. Keeps the inner loop in L1d, removes per-row
  virtual dispatch, gives the compiler and the CPU a chance to pipeline.
- AVX2 intrinsics. Hand-written 256-bit kernels for `int32 < threshold` and
  `int32 SUM`. Eight lanes per vector, packed bitmaps for the filter result.
- Run-length encoding. Compact representation for low-cardinality columns
  with transparent decode-on-the-fly: the source operator yields the same
  4096-row batches whether the column is raw or RLE-encoded.
- Cache-friendly batch sizing. 4096 int32 == 16 KiB == half of a typical L1d
  on Skylake-class cores. Larger batches start spilling to L2.
- Correctness-first SIMD. Every AVX2 operator has a scalar reference
  implementation; a property-style test runs both on random inputs and
  asserts bit-exact-equal output bitmaps and aggregate values.

## Architecture

```
                          query
                            v
   +------------------------+------------------------+
   |  Source     ->  Filter      ->  Aggregate       |
   |  (4096 rows)   (bitmap)        (sum/min/max)    |
   |    raw  or                                      |
   |    RLE decode                                   |
   +-------------------------------------------------+
                            ^
                     CPU feature gate
                            ^
                +-----------+-----------+
                | AVX2 kernel           |
                | (x86_64 + cpuid)      |
                +-----------------------+
                | Scalar fallback       |
                | (everything else)     |
                +-----------------------+
```

Source code layout:

```
src/core      - typed columns, batches, schema
src/rle       - RLE encoder and streaming decoder
src/simd      - cpu_detect + AVX2 / scalar filter+sum kernels
src/exec      - Source / Filter / Aggregate / Sink / Pipeline
src/obs       - latency histogram
tests/unit    - column, RLE, filter, aggregate, pipeline unit tests
tests/fuzz    - libFuzzer harnesses for RLE decode and filter kernel
tests/integration - 5M-row end-to-end query, AVX2-vs-scalar parity
bench         - scan/filter/aggregate microbenchmarks + 100M-row generator
```

## AVX2 filter hot loop (annotated)

```cpp
// data[i] < threshold, packed into one bit per row.
const __m256i thr = _mm256_set1_epi32(threshold);
while (i + 64 <= n) {
    uint64_t bits = 0;
    for (int j = 0; j < 8; ++j) {
        // 8 int32 lanes per load.
        const __m256i v =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + j * 8));
        // _mm256_cmpgt_epi32(thr, v) sets each lane to 0xFFFFFFFF when
        // threshold > data[k], i.e. when data[k] < threshold.
        const __m256i cmp = _mm256_cmpgt_epi32(thr, v);
        // movemask_ps gives us one bit per 32-bit lane: exactly an 8-bit
        // mask covering 8 rows.
        const int mask8 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        bits |= (static_cast<uint64_t>(mask8) & 0xFFull) << (j * 8);
    }
    // 64 rows -> 8 bytes of bitmap.
    std::memcpy(out_bitmap + i / 8, &bits, sizeof(bits));
    i += 64;
}
```

Full source: `src/simd/filter_int32_avx2.cpp`. The scalar mirror is in
`src/simd/filter_int32_scalar.cpp`; both are tested against each other on
1000 random inputs in `tests/unit/filter_test.cpp`.

## Building

Requirements: CMake 3.20+, a C++17 compiler (g++ 11+ or clang 14+),
internet access on first build to fetch GoogleTest.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/columnstore --rows 1000000 --threshold 1000
./build/columnstore --rows 1000000 --threshold 1000 --rle
```

Other modes:

```sh
make asan          # ASan + UBSan build
make scalar        # force scalar path even on x86_64
make fuzz          # libFuzzer harnesses (clang)
make test          # cmake build + ctest
make bench         # run scan_bench, filter_bench, aggregate_bench
```

Docker:

```sh
docker build -t columnstore .
docker run --rm columnstore --rows 1000000 --threshold 1000
```

## Benchmarks

Real numbers from a local run, captured in
[`bench/results/bench_local.json`](bench/results/bench_local.json).
This run was on Apple Silicon (M-series, arm64). The runtime picked the
scalar path because AVX2 is x86-only; clang's auto-vectorizer turned the
scalar sum and filter loops into NEON, so reported throughput is high. The
CI on `ubuntu-22.04` x86_64 runners exercises the hand-written AVX2 path.

| bench           | rows  | path   | best ms | throughput    | p50 ns     | p99 ns     |
| --------------- | ----- | ------ | ------- | ------------- | ---------- | ---------- |
| scan (full pipe)| 100M  | scalar | 74.4    | 1.34 B v/s    | 75,236,041 | 75,638,000 |
| filter kernel   | 100M  | scalar | 42.5    | 2.35 B v/s    | 42,914,875 | 43,146,666 |
| aggregate (sum) | 100M  | scalar | 6.7     | 14.94 B v/s   | 6,724,125  | 6,774,209  |

Honest reading of the 4.1B values/sec claim. The host above did not hit it
on the scan pipeline; the working set is 400 MB of int32 (well outside any
cache), and the loop is dominated by DRAM bandwidth (~10 GB/s effective per
core). The 4.1B/s target is realistic when:

- the input is L1- or L2-resident (a few thousand to a few hundred thousand
  rows, so the bench harness keeps the buffer hot across iterations), and
- AVX2 saturates the load ports on a Skylake-class core.

The aggregate kernel approaches that order at 14.9 B v/s because the
streaming-load + 64-bit-accumulate inner loop pipelines very well; clang
unrolls and vectorizes the scalar version into NEON on ARM, which is why
the AVX2-vs-scalar gap is small on M-series. The CI bench-smoke at 1M rows
asserts a 100M v/s floor.

## What this is not

- No SQL frontend (deferred to v4). The pipeline is built from C++.
- No Parquet or Arrow ingestion (deferred).
- No distributed execution.
- No spilling to disk; columns live in memory.
- No joins. v0 covers Source -> Filter -> Aggregate.
- No AVX-512. AVX2 only.
- No GPU.
- No ARM NEON intrinsics; ARM hosts use the scalar path. clang's
  auto-vectorizer still produces NEON code, but no hand-written NEON kernel
  is shipped.
- No transactions, no MVCC, no concurrent writers. The column is loaded
  once, scanned many times.

## Differences from `bench/results/bench_local.json`

That file is the JSON record of one local run; the README table above is the
human summary of the same numbers. Re-running locally will update both.

## Further reading

- `ARCHITECTURE.md` for the design rationale.
- `docs/execution-model.md` for the vector-at-a-time argument.
- `docs/rle-encoding.md` for the encoding format and when it helps.
- `docs/avx2-operators.md` for the intrinsics walkthrough.
- `docs/scalar-fallback.md` for the runtime feature-detection design.

## License

MIT. See `LICENSE`.
