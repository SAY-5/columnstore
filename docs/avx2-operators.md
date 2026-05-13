# AVX2 operators

Walkthrough of the two intrinsic kernels shipped in v0.

## filter_int32_lt_avx2

`out_bitmap[k]` bit k is set when `data[k] < threshold`. Returns the
total count of set bits.

```cpp
const __m256i thr = _mm256_set1_epi32(threshold);
std::size_t i = 0;

// 64 rows per iteration -> 8 bytes of bitmap per iteration.
while (i + 64 <= n) {
    uint64_t bits = 0;
    for (int j = 0; j < 8; ++j) {
        const __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(data + i + j * 8));
        // We want data < threshold, i.e. threshold > data.
        const __m256i cmp = _mm256_cmpgt_epi32(thr, v);
        // Cast lanes to float (no instruction emitted) so movemask gives
        // one bit per 32-bit lane.
        const int mask8 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        bits |= (static_cast<uint64_t>(mask8) & 0xFFull) << (j * 8);
    }
    std::memcpy(out_bitmap + i / 8, &bits, sizeof(bits));
    i += 64;
}

// 8-row tail.
while (i + 8 <= n) {
    const __m256i v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(data + i));
    const __m256i cmp = _mm256_cmpgt_epi32(thr, v);
    const int mask8 = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
    out_bitmap[i / 8] = static_cast<uint8_t>(mask8 & 0xFF);
    i += 8;
}

// <8 scalar tail.
for (; i < n; ++i) {
    if (data[i] < threshold) {
        out_bitmap[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
}
```

Notes:

- `loadu` not `load`: the runtime buffer may not be 32-byte-aligned. On
  modern x86 cores the unaligned load is the same speed as aligned when the
  address happens to be aligned, so the perf cost is zero for the common
  case.
- `_mm256_cmpgt_epi32` is signed. Using it for `data < threshold` is the
  same as `threshold > data` for signed int32.
- `_mm256_movemask_ps` and `_mm256_castsi256_ps`: the cast is a
  reinterpret-only and emits no instruction; movemask reads the high bit
  of each 32-bit lane. That high bit is set iff the compare wrote
  0xFFFFFFFF, so the 8-bit mask is exactly the predicate result for the 8
  rows.

## sum_int32_avx2

Two paths, depending on whether a selection bitmap is provided.

Unmasked: accumulate 8 lanes of int32 with periodic spills to a 64-bit
accumulator to avoid overflow.

```cpp
__m256i lane_acc = _mm256_setzero_si256();
std::size_t since_flush = 0;
while (i + 8 <= n) {
    const __m256i v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(data + i));
    lane_acc = _mm256_add_epi32(lane_acc, v);
    i += 8;
    since_flush += 8;
    if (since_flush >= 256) {
        // Horizontal reduce + add into the int64 accumulator.
        acc += hsum_epi32(lane_acc);
        lane_acc = _mm256_setzero_si256();
        since_flush = 0;
    }
}
acc += hsum_epi32(lane_acc);
```

Masked: for each 8-element chunk, splat the bitmap byte across 8 lanes and
AND with a per-lane bit pattern to derive a keep mask. Then mask the data
before adding.

```cpp
const __m256i lane_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
while (i + 8 <= n) {
    const __m256i v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(data + i));
    const uint8_t byte = bitmap[i / 8];
    const __m256i splat = _mm256_set1_epi32(byte);
    const __m256i keep_bits = _mm256_and_si256(splat, lane_bits);
    const __m256i keep_mask = _mm256_cmpeq_epi32(keep_bits, lane_bits);
    const __m256i masked = _mm256_and_si256(v, keep_mask);
    lane_acc = _mm256_add_epi32(lane_acc, masked);
    ...
}
```

The horizontal reduction widens to int64 to keep results exact even for
adversarial inputs.

```cpp
int64_t hsum_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m256i lo64 = _mm256_cvtepi32_epi64(lo);
    const __m256i hi64 = _mm256_cvtepi32_epi64(hi);
    const __m256i sum = _mm256_add_epi64(lo64, hi64);
    alignas(32) int64_t out[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(out), sum);
    return out[0] + out[1] + out[2] + out[3];
}
```

## Correctness check

`tests/unit/filter_test.cpp` and `tests/unit/aggregate_test.cpp` each loop
through random sizes and random data, running both the AVX2 and the scalar
kernel, and asserting bit-exact output. Boundary thresholds (INT_MIN,
INT_MAX, 0, +/-1) are tested explicitly. Tail sizes (1, 7, 8, 9, 15, 16,
63, 64, 65, 71, 4095, 4096, 4097) are tested to cover the three-region
loop structure.

## Why not AVX-512?

- AVX-512 is not universal on commodity x86 in 2026 server fleets,
  especially after the Alder Lake / Sapphire Rapids consumer split. Going
  AVX2 keeps the binary portable to the widest ubuntu-22.04 hosts.
- AVX-512 introduces a non-trivial frequency-throttling problem on some
  Skylake-X cores. For a single-operator hot path this is fine; for a
  pipeline it can hurt the un-vectorized parts of the program.
- AVX2 is the sweet spot for this learning project. Adding AVX-512 once
  AVX2 works is a follow-up.
