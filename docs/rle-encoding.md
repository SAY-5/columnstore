# RLE encoding

Run-length encoding represents a sequence of equal values as a single
(value, length) pair.

## Format

Two parallel `std::vector`s:

- `values`: int32 (or whatever the column type is)
- `lengths`: uint32, always > 0

A column of N rows expands to `K` (value, length) pairs where K is the
number of distinct adjacent values. Sum of lengths == N.

## API

```cpp
RleEncodedInt32 enc = rle_encode_int32(data.data(), n);
Column<int32_t> col = Column<int32_t>::from_rle(std::move(enc.values),
                                                std::move(enc.lengths));
```

For decoding, the source operator wraps a streaming `RleDecoder`:

```cpp
auto scan = std::make_unique<Int32ColumnScan>(
    col.rle_values().data(), col.rle_lengths().data(),
    col.rle_values().size());
```

The decoder yields up to `kBatchSize` values per `next_batch` call. The
result lives in a stack buffer; the source forwards it downstream without
copying.

## When to use RLE

Good fit:

- Sorted columns: a primary-key-ordered column has very long runs after a
  group-by-like key.
- Low-cardinality categorical columns: `state IN ('NY', 'CA', ...)`.
- Bitfield-style columns: lots of zeros punctuated by an occasional one.

Bad fit:

- Random or near-unique columns. Average run length is ~1, so each value
  occupies 8 bytes instead of 4. Decoder overhead is wasted.
- Floating-point columns where adjacent equal values are rare.

## Malformed input

The streaming decoder treats a zero-length run (except the trivial "empty
column" case) as malformed and returns false from `next_batch`. The
fuzzer harness in `tests/fuzz/rle_fuzz.cpp` drives the decoder with random
bytes and asserts no crash.

## Future work

- Bit-packed integer encoding for the dense regions.
- Frame-of-reference + delta + bit-packing (Parquet's RLE_DICTIONARY).
- Dictionary encoding for string columns.

None of these are in v0; the goal here is to study the simplest encoding
that interacts non-trivially with the SIMD scan path.
