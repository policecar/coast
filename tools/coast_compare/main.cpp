#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "compare.h"

namespace {

void print_usage(std::ostream& out, const char* prog) {
    out << "Usage: " << prog
        << " <candidate.json> <goldens-dir/> [--tolerances <path>] [--aggregate] [--verbose]\n\n";
    out << "  --aggregate  compare numeric arrays by their (mean, max, std) summary\n";
    out << "               instead of element-wise (uses the 'aggregate' tolerance block)\n\n";
    out << "Exit codes:\n";
    out << "  0  matched at least one retained golden within tolerance\n";
    out << "  1  no retained golden matched\n";
    out << "  2  structural error (missing field, dtype/shape mismatch,\n";
    out << "     missing binary ref, NaN, empty goldens dir, etc.)\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string candidate;
    std::string goldens_dir;
    std::string tolerances_path;
    bool        verbose = false;
    auto        mode    = coast::compare::compare_mode::strict;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verbose" || a == "-v") {
            verbose = true;
        } else if (a == "--aggregate") {
            mode = coast::compare::compare_mode::aggregate;
        } else if (a == "--tolerances") {
            if (i + 1 >= argc) {
                std::cerr << "--tolerances requires a path argument\n";
                return 2;
            }
            tolerances_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(std::cout, argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(std::cerr, argv[0]);
            return 2;
        } else if (candidate.empty()) {
            candidate = std::move(a);
        } else if (goldens_dir.empty()) {
            goldens_dir = std::move(a);
        } else {
            std::cerr << "Unexpected positional argument: " << a << "\n";
            return 2;
        }
    }

    if (candidate.empty() || goldens_dir.empty()) {
        print_usage(std::cerr, argv[0]);
        return 2;
    }

    coast::compare::tolerance_set tols;
    if (!tolerances_path.empty()) {
        try {
            tols = coast::compare::tolerance_set::load(tolerances_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load tolerances '" << tolerances_path
                      << "': " << e.what() << "\n";
            return 2;
        }
    }

    auto outcome = coast::compare::compare_against_goldens(
        candidate, goldens_dir, tols, mode);
    coast::compare::render_report(std::cout, outcome, verbose);
    return coast::compare::exit_code_of(outcome);
}
