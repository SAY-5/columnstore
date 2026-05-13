#include "exec/sink.h"

namespace columnstore {

Int32Sink::Int32Sink(std::unique_ptr<Int32Operator> child) : child_(std::move(child)) {}

std::vector<int32_t> Int32Sink::collect() {
    std::vector<int32_t> out;
    while (true) {
        auto in = child_->next();
        if (!in) {
            break;
        }
        if (!in->has_selection) {
            out.insert(out.end(), in->data, in->data + in->size);
            continue;
        }
        const uint8_t* bm = in->selection.data();
        for (std::size_t i = 0; i < in->size; ++i) {
            const bool keep = (bm[i / 8] >> (i % 8)) & 1u;
            if (keep) {
                out.push_back(in->data[i]);
            }
        }
    }
    return out;
}

} // namespace columnstore
