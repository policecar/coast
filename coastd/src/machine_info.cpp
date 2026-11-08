#include "daemon/machine_info.h"

#include <unistd.h>   // gethostname

#include <filesystem>
#include <system_error>

#include "daemon/child_cmd.h"   // dataset_link_target

namespace coastd {

namespace {

std::string os_name() {
#if defined(__linux__)
    return "linux";
#elif defined(__APPLE__)
    return "macos";
#else
    return "unknown";
#endif
}

std::string hostname() {
    char buf[256];
    if (::gethostname(buf, sizeof buf) == 0) { buf[sizeof buf - 1] = '\0'; return buf; }
    return "unknown";
}

} // namespace

proto::machine_summary make_machine_summary(const daemon_config& cfg, std::string_view version) {
    proto::machine_summary m;
    m.hostname       = hostname();
    m.os             = os_name();
    m.coastd_version = std::string(version);
    m.proto_version  = proto::proto_version;   // advertised pre-submit (protocol.h)
    m.slots          = cfg.slots;
    m.runs_root      = cfg.runs_root.string();
    m.repo_root      = cfg.repo_root.string();
    m.dataset_base   = dataset_link_target(cfg.dataset_base).string();   // the `data` link's target
    m.tcp_port       = cfg.tcp_port;   // so a discovered daemon is connectable
    // Will a config's relative "data/…" path resolve in a run dir here? Only if
    // dataset_base is set AND its target exists — a configured-but-missing target
    // produces a dangling link, i.e. a job that still fails at dataset load, so
    // answering yes on configuration alone would advertise a promise the daemon
    // cannot keep. Checked per call rather than cached: the machine description is
    // otherwise static, but this one bit changes when the operator mounts the
    // share, and a stale "no" would outlive the fix for the daemon's lifetime.
    std::error_code ec;
    m.data_link_ok = !cfg.dataset_base.empty() && std::filesystem::exists(cfg.dataset_base, ec);
    return m;
}

} // namespace coastd
