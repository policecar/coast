#include "h5browse/row_filter.h"

#include <format>

namespace h5b {

std::string row_filter::describe() const
{
    if (empty()) return "unfiltered";
    std::string s;
    if (row_begin || row_end) {
        s = std::format("rows [{},{})",
                        row_begin ? std::format("{}", *row_begin) : "0",
                        row_end ? std::format("{}", *row_end) : "end");
    }
    for (const column_condition& c : conditions) {
        if (!s.empty()) s += " & ";
        if (c.op == filter_op::is_nan || c.op == filter_op::is_inf ||
            c.op == filter_op::is_finite) {
            s += std::format("col {} {}", c.column, filter_op_name(c.op));
        } else {
            s += std::format("col {} {} {}", c.column, filter_op_name(c.op), c.value);
        }
    }
    return s;
}

} // namespace h5b
