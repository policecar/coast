//
// Test helper: captures diag output for the lifetime of a scope.
//
// The destructor restores the default sink and level, so a failed REQUIRE or a
// throwing call inside the scope cannot leave the sink writing into a destroyed
// object. Each record keeps its level and category, so a test can check the
// messages of one category only.
//

#ifndef COAST_TEST_DIAG_CAPTURE_H
#define COAST_TEST_DIAG_CAPTURE_H

#include <string>
#include <string_view>
#include <vector>

#include "diag/diag.h"

namespace coast::test {

struct diag_capture {
    struct record_t {
        coast::diag::level lvl;
        std::string        cat, msg;
    };
    std::vector<record_t> got;

    diag_capture()
    {
        coast::diag::set_level(coast::diag::level::info);
        coast::diag::set_sink([this](coast::diag::level l, std::string_view c, std::string_view m) {
            got.push_back({ l, std::string(c), std::string(m) });
        });
    }
    ~diag_capture() { coast::diag::set_sink({}); coast::diag::set_level(coast::diag::level::info); }
    diag_capture(const diag_capture&)            = delete;
    diag_capture& operator=(const diag_capture&) = delete;

    // The messages of the warnings in `category`, in emission order.
    [[nodiscard]] std::vector<std::string> warnings(std::string_view category) const
    {
        std::vector<std::string> out;
        for (const auto& r : got)
            if (r.lvl == coast::diag::level::warn && r.cat == category) out.push_back(r.msg);
        return out;
    }
};

}

#endif // COAST_TEST_DIAG_CAPTURE_H
