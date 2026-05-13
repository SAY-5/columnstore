# Execution model

This engine runs vector-at-a-time. Operators move 4096-value batches at a
step, not one row at a time. This file walks through why that shape was
chosen.

## Operator interface

```cpp
class Int32Operator {
public:
    virtual std::optional<Batch<int32_t>> next() = 0;
};
```

Pull-based: a consumer calls `child->next()` until it gets `std::nullopt`.
Each call returns up to `kBatchSize = 4096` rows. The selection bitmap on
the batch carries the per-row predicate result so downstream operators can
skip masked-out rows.

## Why 4096 rows

The size is a balance:

- L1d budget. Skylake-class cores have 32 KiB of L1d per core. Two batches
  of int32 plus a small bitmap fit, with room for the operator's working
  state.
- Auto-vectorization. The vectorizer benefits from straight-line, fixed-
  bound loops. 4096 is large enough that the vectorized tail is a small
  fraction of total work.
- Operator overhead. The cost of a `next()` call is amortized over 4096
  rows. At ns-per-row scan times, that overhead is irrelevant.

If a column is shorter than 4096, the source returns a partial batch and
then EOF; nothing about the operator interface depends on the batch being
exactly full.

## Push-based vs pull-based

This engine is pull-based. The sink drives the pipeline by repeatedly
asking its child for a batch. Two consequences:

- LIMIT-style operators can short-circuit without waking the source.
- Each batch lives on the caller's stack frame; there is no inter-operator
  queue, no allocation per batch, no synchronization.

The trade-off is that a pull-based pipeline can't easily exploit operator
parallelism (one thread per operator). That is a deferred concern; the v0
target is a single-core hot path.

## Row-at-a-time, for contrast

The naive alternative looks like:

```cpp
for (int32_t v : column) {
    if (v > threshold) {
        sum += v;
    }
}
```

What goes wrong:

- The branch is data-dependent, so the predictor mostly fails on random
  inputs.
- Even if it predicted, the loop runs one int32 per iteration. AVX2 wants
  eight.
- Wrapping each row in a virtual call (the "iterator" pattern) adds a
  pointer indirection per row.

The vector-at-a-time loop replaces the row-by-row branch with a vector
compare that produces a bitmap. The bitmap is consumed by the aggregate,
which then runs a branch-free masked sum. Both kernels run at near
peak throughput when the input is L1-resident.
