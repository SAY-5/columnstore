#include "exec/filter.h"

#include "simd/cpu_detect.h"
#include "simd/kernels.h"

namespace columnstore {

Int32Filter::Int32Filter(std::unique_ptr<Int32Operator> child, FilterOp op, int32_t threshold)
    : child_(std::move(child)), op_(op), threshold_(threshold) {}

std::optional<Batch<int32_t>> Int32Filter::next() {
    auto in = child_->next();
    if (!in) {
        return std::nullopt;
    }
    Batch<int32_t> out;
    out.data = in->data;
    out.size = in->size;
    out.has_selection = true;
    out.selection.assign(Batch<int32_t>::bitmap_bytes(in->size), 0);

    std::size_t passed = 0;
    const SimdPath path = active_simd_path();
    if (path == SimdPath::Avx2) {
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
        if (op_ == FilterOp::LessThan) {
            passed = filter_int32_lt_avx2(in->data, in->size, threshold_, out.selection.data());
        } else {
            passed = filter_int32_gt_avx2(in->data, in->size, threshold_, out.selection.data());
        }
#else
        // Compiled without AVX2 -> fall through to scalar at runtime.
        if (op_ == FilterOp::LessThan) {
            passed = filter_int32_lt_scalar(in->data, in->size, threshold_, out.selection.data());
        } else {
            passed = filter_int32_gt_scalar(in->data, in->size, threshold_, out.selection.data());
        }
#endif
    } else {
        if (op_ == FilterOp::LessThan) {
            passed = filter_int32_lt_scalar(in->data, in->size, threshold_, out.selection.data());
        } else {
            passed = filter_int32_gt_scalar(in->data, in->size, threshold_, out.selection.data());
        }
    }
    rows_seen_ += in->size;
    rows_passed_ += passed;
    return out;
}

} // namespace columnstore
