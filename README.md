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
1000 random inputs in `tests/unit/filter_test.cpp` and on a wider
cross-product (1000+ random `(values, threshold)` pairs per operator,
boundary thresholds INT_MIN/-1/0/1/INT_MAX, fixed bit patterns
0x00/0xFF/0xAA/0x55, tail sizes 1..71 and 4090..4097) in
`tests/unit/property_test.cpp`. The fuzz harnesses run 10000 iterations
per harness (rle, filter) on every CI build.

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

Real numbers, captured in
[`bench/results/bench_local.json`](bench/results/bench_local.json).

### x86_64 (GitHub Actions ubuntu-22.04, 1M rows, cache-resident)

This is the CI `bench-smoke` job; numbers come from the actual run that
gates this README. The AVX2 path is the hand-written intrinsic kernel.

| bench           | rows | path   | best ms | throughput    | p50 ns  |
| --------------- | ---- | ------ | ------- | ------------- | ------- |
| scan (full pipe)| 1M   | avx2   | 0.762   | 1.312 B v/s   | 778,432 |
| filter kernel   | 1M   | avx2   | 0.128   | 7.788 B v/s   | 128,820 |
| filter kernel   | 1M   | scalar | 1.015   | 0.985 B v/s   | 1,016,364 |
| aggregate (sum) | 1M   | avx2   | 0.133   | 7.539 B v/s   | 133,047 |
| aggregate (sum) | 1M   | scalar | 0.175   | 5.722 B v/s   | 174,759 |

AVX2 vs scalar speedup: filter 7.9x, aggregate 1.32x. (The aggregate gap
is small because `gcc -O3` auto-vectorizes the scalar sum loop into AVX2;
the filter loop has a bitmap-pack step the auto-vectorizer can't
synthesize from straight-line code.)

### arm64 (Apple Silicon, 100M rows, DRAM-bound)

The runtime picked the scalar path because AVX2 is x86-only. Clang's
auto-vectorizer turned the scalar sum and filter into NEON, so reported
throughput is high; this isn't the hand-written intrinsic path.

| bench           | rows  | path   | best ms | throughput    | p50 ns     |
| --------------- | ----- | ------ | ------- | ------------- | ---------- |
| scan (full pipe)| 100M  | scalar | 74.4    | 1.344 B v/s   | 75,236,041 |
| filter kernel   | 100M  | scalar | 42.5    | 2.352 B v/s   | 42,914,875 |
| aggregate (sum) | 100M  | scalar | 6.7     | 14.94 B v/s   | 6,724,125  |

Honest reading of the 4.1B values/sec claim:

- The filter kernel hits 7.788 B v/s on x86_64 at 1M rows (cache-resident),
  exceeding the 4.1 B/s target by 1.9x.
- The aggregate kernel hits 7.539 B v/s under the same conditions.
- The full-pipeline scan does not hit it. Per-batch bitmap allocation and
  pipeline plumbing impose overhead the raw kernels don't see; at 1M rows
  the pipeline runs at 1.31 B v/s.
- At 100M rows (400 MB of int32, well outside any cache) DRAM bandwidth
  is the bottleneck on every architecture and throughput drops to
  ~1.3 B v/s.

CI `bench-smoke` asserts a 100M v/s floor and currently exceeds it by 13x.

### Scaling table (1M / 10M / 100M)

Captured on the local ARM dev host with the scalar path active; the layout is
identical on x86_64 CI.  Files: `bench/results/local_1m.jsonl`,
`local_10m.jsonl`, `local_100m.jsonl`. CI emits the same fields per run via
`COLUMNSTORE_BENCH_JSON_OUT` and the `bench-regress` job compares against
`bench/results/ci_baseline.jsonl` with a 30% drift threshold on both
throughput and P99.

| operator         | 1M B v/s | 10M B v/s | 100M B v/s | 100M P99 ns |
| ---------------- | -------- | --------- | ---------- | ----------- |
| filter_gt_avx2   | 2.338    | 2.303     | 2.361      | 44,006,125  |
| filter_gt_scalar | 2.401    | 2.292     | 2.387      | 42,143,334  |
| sum_avx2         | 21.39    | 14.82     | 13.36      | 7,890,125   |
| sum_scalar       | 21.30    | 14.14     | 14.61      | 7,626,375   |
| scan (pipeline)  | 1.356    | 1.307     | 1.370      | 76,313,500  |

Throughput peaks at 1M (cache-resident) and degrades into DRAM-bound territory
above L2 working sets, exactly as the kernel-vs-pipeline split predicts. The
filter kernel is bandwidth-bound on this host across all three sizes; the sum
kernel is compute-bound at 1M and bandwidth-bound at 100M.

### Cache-line padding study

`bench/padding_bench.cpp` measures whether forcing 64-byte alignment on the
per-column descriptor (Batch/Column metadata) moves the needle on a tight
multi-column scan loop. Result on the local host
(`bench/results/padding_local.jsonl`):

| layout            | sizeof | 12.8M-row best ms | B v/s |
| ----------------- | ------ | ----------------- | ----- |
| default-aligned   | 16     | 32.58             | 0.393 |
| 64B-aligned       | 64     | 31.48             | 0.407 |

A ~3% improvement, within noise. Quadrupling per-column metadata size to pad
to a cache line is not worth it for this workload; the data path (the actual
column buffers) already dominates the loads.

## Dictionary encoding (low-cardinality columns)

`DictColumn<T>` is a second compression mode (alongside RLE) for columns whose
distinct value count is small. When `cardinality <= 256` the column stores:

- `dict`: a small contiguous array of the distinct values, in insertion order
- `codes`: a `uint8_t` buffer with one byte per row, indexing into `dict`

This pays off in two places:

1. `CountDistinct` is O(K) instead of O(N). On a cardinality-8 column with
   1M rows, the dict path runs in nanoseconds (it just reports the dict
   size) while the unordered_set scalar fallback takes ~6 ms. The local
   measurement in `dict_test.cpp` shows the dict path at effectively 0 ns
   versus 5.99 ms for the scalar fallback (>>1000x).
2. Equality filter on dict codes uses `_mm256_cmpeq_epi8` on the byte
   buffer, which is 4x the throughput of the int32 `_mm256_cmpeq_epi32`
   path because each instruction processes 32 lanes instead of 8.

The codes-buffer also halves to a quarter of the original column size
(4 bytes -> 1 byte per row) on int32 columns, and is even larger savings
on string columns where each row was a pointer + length pair.

`DictColumn::try_build(raw)` returns `std::nullopt` when cardinality
exceeds the byte-coded ceiling. The encoder is one pass over the data;
the runtime cost is amortized over every subsequent scan/filter/count.

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
