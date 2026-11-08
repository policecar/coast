#ifndef COAST_TEST_SUBPROCESS_H
#define COAST_TEST_SUBPROCESS_H

#include <map>
#include <string>
#include <vector>

namespace coast::test {

struct subprocess_result {
    int         exit_code = -1;     // -1 if process didn't terminate normally
    std::string stdout_text;
    std::string stderr_text;
};

// Run an executable with the given argv (excluding argv[0], which is set to
// exe). stdout and stderr are captured separately, fully buffered. Extra
// environment variables are added on top of the inherited environment.
//
// cwd: if non-empty, child runs with that working directory.
//
// Throws std::runtime_error on fork/exec/pipe failure (a test-infrastructure
// problem, not a test failure).
subprocess_result run(const std::string&              exe,
                      const std::vector<std::string>& args,
                      const std::map<std::string, std::string>& extra_env = {},
                      const std::string&              cwd = {});

} // namespace coast::test

#endif // COAST_TEST_SUBPROCESS_H
