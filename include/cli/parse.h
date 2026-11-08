#ifndef COAST_CLI_PARSE_H
#define COAST_CLI_PARSE_H

#include <iosfwd>
#include <stdexcept>

#include "cli/options.h"

namespace coast::cli {

struct parse_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The one wording for "a headless run needs a length", shared by cli::parse (the
// explicit -H case) and main() (the implicit promotion in a COAST_WITH_GUI=OFF
// build), so a user meets the same sentence whichever path rejects the command.
inline constexpr const char* headless_needs_length_msg =
    "A headless run needs an explicit length: pass -n/--steps <count> to run a "
    "fixed number of steps, or --unlimited to run until interrupted.";

// Parse argv into an options struct. Throws parse_error on malformed input.
//
// Build-independent by design: THIS FUNCTION behaves identically in a GUI and a
// headless build, so its unit tests mean the same thing in both. The rule that
// does vary — a headless build promoting every run to headless, and therefore
// demanding a run length even without -H — lives in main() instead, sharing the
// message below.
//
// The claim is deliberately scoped to parse(), not to this header: print_usage()
// below IS build-dependent (it #ifdefs COAST_WITH_GUI to describe the right flag
// set), which is fine — it renders text and parses nothing.
options parse(int argc, char** argv);

// Print usage to the given stream. Build-DEPENDENT: the wording and the set of
// flags shown differ between a GUI and a headless build.
void print_usage(std::ostream& out, const char* program_name);

} // namespace coast::cli

#endif // COAST_CLI_PARSE_H
