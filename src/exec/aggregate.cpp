#include "exec/aggregate.h"

#include "simd/cpu_detect.h"
#include "simd/kernels.h"

namespace columnstore {

Int32Aggregate::Int32Aggregate(std::unique_ptr<Int32Operator> child, AggregateKind kind)
    : child_(std::move(child)), kind_(kind) {}

AggregateResult Int32Aggregate::run() {
    AggregateResult r;
    r.kind = kind_;
    const SimdPath path = active_simd_path();

    while (true) {
        auto in = child_->next();
        if (!in) {
            break;
        }
        const uint8_t* bm = in->has_selection ? in->selection.data() : nullptr;

        if (kind_ == AggregateKind::Sum || kind_ == AggregateKind::Avg) {
            int64_t s = 0;
            if (path == SimdPath::Avx2) {
#if defined(COLUMNSTORE_COMPILED_WITH_AVX2)
                s = sum_int32_avx2(in->data, in->size, bm);
#else
                s = sum_int32_scalar(in->data, in->size, bm);
#endif
            } else {
                s = sum_int32_scalar(in->data, in->size, bm);
            }
            r.sum += s;
            r.count += count_int32_scalar(in->size, bm);
            if (r.count > 0) {
                r.empty = false;
            }
        } else if (kind_ == AggregateKind::Count) {
            r.count += count_int32_scalar(in->size, bm);
            if (r.count > 0) {
                r.empty = false;
            }
        } else if (kind_ == AggregateKind::Min) {
            bool empty = true;
            const int32_t m = min_int32_scalar(in->data, in->size, bm, &empty);
            if (!empty) {
                if (r.empty) {
                    r.min = m;
                    r.empty = false;
                } else if (m < r.min) {
                    r.min = m;
                }
            }
        } else if (kind_ == AggregateKind::Max) {
            bool empty = true;
            const int32_t m = max_int32_scalar(in->data, in->size, bm, &empty);
            if (!empty) {
                if (r.empty) {
                    r.max = m;
                    r.empty = false;
                } else if (m > r.max) {
                    r.max = m;
                }
            }
        }
    }
    return r;
}

} // namespace columnstore
