#include "core/schema.h"

namespace columnstore {

int Schema::index_of(const std::string& name) const {
    for (std::size_t i = 0; i < cols_.size(); ++i) {
        if (cols_[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace columnstore
