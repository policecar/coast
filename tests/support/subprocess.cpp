#include "subprocess.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace coast::test {

namespace {

[[noreturn]] void die(const std::string& what) {
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

// Build a NULL-terminated char* array from std::strings. The vector itself
// must outlive the returned pointers.
std::vector<char*> as_argv(const std::string& exe,
                           const std::vector<std::string>& args,
                           std::vector<std::string>& storage) {
    storage.reserve(args.size() + 1);
    storage.push_back(exe);
    for (const auto& a : args) storage.push_back(a);
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& s : storage) argv.push_back(s.data());
    argv.push_back(nullptr);
    return argv;
}

// Same trick for envp: inherit existing env, then overwrite/append from extras.
std::vector<char*> build_envp(const std::map<std::string, std::string>& extras,
                              std::vector<std::string>& storage) {
    std::map<std::string, std::string> merged;
    for (char** e = environ; e && *e; ++e) {
        const std::string entry = *e;
        const auto eq = entry.find('=');
        if (eq == std::string::npos) continue;
        merged[entry.substr(0, eq)] = entry.substr(eq + 1);
    }
    for (const auto& [k, v] : extras) merged[k] = v;

    storage.reserve(merged.size());
    for (const auto& [k, v] : merged) storage.push_back(k + "=" + v);

    std::vector<char*> envp;
    envp.reserve(storage.size() + 1);
    for (auto& s : storage) envp.push_back(s.data());
    envp.push_back(nullptr);
    return envp;
}

} // namespace

subprocess_result run(const std::string& exe,
                      const std::vector<std::string>& args,
                      const std::map<std::string, std::string>& extra_env,
                      const std::string& cwd) {
    int out_pipe[2];
    int err_pipe[2];
    if (::pipe(out_pipe) != 0) die("pipe(stdout)");
    if (::pipe(err_pipe) != 0) die("pipe(stderr)");

    const pid_t pid = ::fork();
    if (pid < 0) die("fork");

    if (pid == 0) {
        // --- child ---
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);

        if (!cwd.empty() && ::chdir(cwd.c_str()) != 0) {
            std::fprintf(stderr, "subprocess: chdir(%s) failed: %s\n",
                         cwd.c_str(), std::strerror(errno));
            std::_Exit(127);
        }

        std::vector<std::string> argv_storage;
        std::vector<std::string> envp_storage;
        auto argv = as_argv(exe, args, argv_storage);
        auto envp = build_envp(extra_env, envp_storage);

        ::execve(exe.c_str(), argv.data(), envp.data());
        std::fprintf(stderr, "subprocess: execve(%s) failed: %s\n",
                     exe.c_str(), std::strerror(errno));
        std::_Exit(127);
    }

    // --- parent ---
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    // Drain stdout and stderr concurrently to avoid the child blocking on a
    // full pipe (in case it writes a lot before exiting).
    subprocess_result res;
    std::array<pollfd, 2> pfds{
        pollfd{out_pipe[0], POLLIN, 0},
        pollfd{err_pipe[0], POLLIN, 0}
    };
    std::array<std::string*, 2> sinks{ &res.stdout_text, &res.stderr_text };
    std::array<bool, 2>         open { true, true };

    while (open[0] || open[1]) {
        // Refresh fds (closed ones get -1 so poll ignores them).
        for (int i = 0; i < 2; ++i) {
            pfds[i].fd      = open[i] ? pfds[i].fd : -1;
            pfds[i].revents = 0;
        }
        int rc = ::poll(pfds.data(), pfds.size(), -1);
        if (rc < 0) {
            if (errno == EINTR) continue;
            die("poll");
        }
        for (int i = 0; i < 2; ++i) {
            if (!open[i]) continue;
            if (pfds[i].revents & (POLLIN | POLLHUP)) {
                std::array<char, 4096> buf{};
                ssize_t n = ::read(pfds[i].fd, buf.data(), buf.size());
                if (n > 0) {
                    sinks[i]->append(buf.data(), static_cast<std::size_t>(n));
                } else if (n == 0) {
                    ::close(pfds[i].fd);
                    open[i] = false;
                } else if (errno != EINTR) {
                    die("read");
                }
            }
        }
    }

    int status = 0;
    while (true) {
        const pid_t w = ::waitpid(pid, &status, 0);
        if (w == pid) break;
        if (w < 0 && errno == EINTR) continue;
        die("waitpid");
    }
    if (WIFEXITED(status)) res.exit_code = WEXITSTATUS(status);
    return res;
}

} // namespace coast::test
