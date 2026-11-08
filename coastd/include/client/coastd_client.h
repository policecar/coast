#ifndef COAST_COASTD_CLIENT_H
#define COAST_COASTD_CLIENT_H

// -----------------------------------------------------------------------------
// coastd_client — the THIN client side of the coastd wire protocol.
//
// This is what a frontend (the REPL console, the GUI) links to talk to a daemon.
// It depends ONLY on coastd_proto (the header contract + nlohmann/json): NO
// coast_lib, NO coastd_core, NO simulator. A `connection` is one blocking TCP
// socket; each call is a single synchronous JSONL round-trip (write one request
// line, read one reply line). There is no client-side subscriber state — a
// frontend stays fresh by POLLING list() on a timer (~2 s), exactly as the
// design's stateless-resync model prescribes.
//
// Errors are returned, never thrown: every call takes a `std::string& err` it
// fills on failure (transport error, or a daemon {"ok":false,"error":...}). A
// nullopt / false return means "see err".
// -----------------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "daemon/protocol.h"

namespace coastd::client {

struct endpoint {
    std::string   host = "localhost";
    std::uint16_t port = proto::default_tcp_port;
};

// -----------------------------------------------------------------------------
// Fixed-list discovery (IDD §6.6, the DETERMINISTIC PRIMARY): a client-side
// config file naming known daemons, so a frontend reaches "gpu-box" by name with
// no broadcast at all. UDP discover() is the soft convenience layer on top; this
// is the reliable floor. `localhost` always resolves, even with no config file.
// -----------------------------------------------------------------------------

// One entry of the fixed-list config.
struct host_entry {
    std::string   name;                            // human handle, e.g. "gpu-box"
    std::string   host;                            // hostname or IP
    std::uint16_t port = proto::default_tcp_port;  // command port
};

class host_config {
public:
    // Load from a JSON file: [{"name","host","port"?}, …]. A MISSING file is not an
    // error — it yields a config with just the built-in `localhost` entry. If the
    // file exists but is malformed, `warn` is filled with a human note and the
    // localhost default is still returned (a frontend stays usable). A `localhost`
    // entry is always present (synthesized if the file omits one).
    static host_config load(const std::filesystem::path& path, std::string& warn);

    // Load from the default location: $COASTD_HOSTS, else
    // $XDG_CONFIG_HOME/coastd/hosts.json, else ~/.config/coastd/hosts.json.
    static host_config load_default(std::string& warn);

    // Resolve a name to an endpoint. An empty name resolves to `localhost`.
    // nullopt if the name is not in the list.
    [[nodiscard]] std::optional<endpoint> resolve(std::string_view name) const;

    [[nodiscard]] const std::vector<host_entry>& entries() const { return entries_; }

    // The default config path (respecting $COASTD_HOSTS / $XDG_CONFIG_HOME / HOME).
    static std::filesystem::path default_path();

private:
    std::vector<host_entry> entries_;
};

// Arguments for a submit. Mirrors the server's job_spec fields that a client sets.
struct submit_args {
    std::string   ref;            // required: branch/tag/sha to build
    // required: the config's CONTENT, not a path. The daemon may be a different
    // machine, where a path either does not exist or — worse — exists with
    // different content and silently runs the wrong config (protocol.h). Fill these
    // two with load_config_file() rather than by hand.
    std::string   config_text;    // required: raw bytes of the config
    std::string   config_name;    // optional: basename, used only for the job mnemonic
    std::string   label;          // optional mnemonic
    std::string   message;        // optional free-text note
    std::uint64_t steps_target = 0;
    int           priority     = 0;
};

// Read a config from THIS machine into `a` (config_text + config_name).
//
// Pre-flight-parses the file so a malformed or non-UTF-8 config is reported here,
// with a clear message and no round trip, instead of surfacing as a raw
// nlohmann type_error.316 ("invalid UTF-8 byte at index N") when the request is
// serialized. The daemon re-validates authoritatively regardless — this is fast
// feedback, never a substitute for the server's own checks.
//
// The ORIGINAL bytes are stored, never a re-dump of the parsed value: the daemon
// writes them verbatim as the run's provenance record, and a re-dump would reorder
// keys and drop the file's formatting.
[[nodiscard]] bool load_config_file(submit_args& a, const std::filesystem::path& local_path,
                                    std::string& err);

class connection {
public:
    // Connect to `ep` within `connect_timeout`. nullopt on failure (err filled).
    //
    // `io_timeout` bounds each SUBSEQUENT request/reply instead; zero (the default)
    // means "same as connect_timeout". They are separate because they answer
    // different questions. A connect bound must be SHORT when the caller is a UI
    // thread — it is how long a dead host may freeze the frame. An operation bound
    // must cover the daemon's slowest legitimate reply: `submit` blocks on the
    // core's single-writer queue and two fsyncs, so on a busy disk it is far slower
    // than a TCP handshake. Sharing one value forces a choice between a frozen UI
    // and reporting a SUCCESSFUL submit as a failure — and the natural user
    // response to that false failure is to submit again, creating a duplicate job.
    static std::optional<connection> open(const endpoint& ep,
                                          std::chrono::milliseconds connect_timeout,
                                          std::string& err,
                                          std::chrono::milliseconds io_timeout = {});

    connection(connection&&) noexcept;
    connection& operator=(connection&&) noexcept;
    connection(const connection&)            = delete;
    connection& operator=(const connection&) = delete;
    ~connection();

    // One round-trip each. nullopt/false => err set (transport or daemon error).
    std::optional<std::vector<proto::job_summary>> list(std::string& err);
    std::optional<std::string>            submit(const submit_args& a, std::string& err);  // -> job id
    bool                                  cancel(const std::string& id, std::string& err);
    bool                                  reorder(const std::string& id, int pos, std::string& err);
    // Move relative to the job's CURRENT position (negative = towards the front),
    // resolved by the daemon against the live queue. Prefer this over reorder()
    // wherever the position came from a polled snapshot — see the daemon-side note.
    bool                                  reorder_by(const std::string& id, int delta, std::string& err);
    std::optional<proto::machine_summary> machine_info(std::string& err);
    // SHA-256 of files under the daemon's dataset_base, paths relative to it.
    std::optional<std::map<std::string, std::string>>
    hash_inputs(const std::vector<std::string>& paths, std::string& err);

    // Is the socket still usable? A transport error (timed-out/failed round-trip)
    // POISONS the connection — see round_trip. A poisoned connection returns an
    // error from every subsequent call; a frontend must drop it and reconnect.
    [[nodiscard]] bool alive() const noexcept { return fd_ >= 0; }

    // The whole-reply bound each round-trip uses (read-only observation point).
    // It must survive the move out of open() — a dropped one silently became 0 =
    // unbounded — and a test checks exactly that here, without timing a reply.
    [[nodiscard]] std::chrono::milliseconds io_timeout() const noexcept { return timeout_; }

private:
    explicit connection(int fd, std::chrono::milliseconds timeout)
        : fd_(fd), timeout_(timeout) {}
    // Send {cmd, ...args} as one line, read one reply line. nullopt on transport
    // failure; otherwise the parsed reply object (which may be {"ok":false,...}).
    std::optional<nlohmann::json> round_trip(std::string_view cmd, nlohmann::json args,
                                             std::string& err);
    // Close the socket and mark the connection dead. Called on ANY transport error:
    // a timed-out reply is still in flight, so the next round-trip on this socket
    // would read the PREVIOUS command's reply (the protocol has no request-id to
    // detect the desync). Poisoning turns that latent mis-attribution into a clean
    // "connection closed" the frontend recovers from by reconnecting.
    void poison() noexcept;

    int         fd_ = -1;
    std::string rbuf_;   // carry-over bytes past a line boundary
    // The caller's bound, retained so a reply has a WHOLE-reply deadline and not
    // merely a per-recv one. SO_RCVTIMEO restarts on every arriving byte, so a peer
    // dribbling one byte per window holds the caller for max_line_bytes × timeout —
    // on the GUI that is the render thread. See read_line.
    std::chrono::milliseconds timeout_{0};
};

// UDP Model-P discovery: broadcast a probe and collect every daemon that answers
// within `timeout`. Best-effort convenience on top of a fixed host list.
std::vector<proto::machine_summary> discover(std::chrono::milliseconds timeout,
                                             std::uint16_t port = proto::default_udp_port);

// Resolve `name` against the DEFAULT fixed-list config and connect. nullopt if the
// name is unknown (err says so) or the connect fails (err filled). An empty name
// connects to localhost. This is the deterministic connect-by-name path.
std::optional<connection> connect_to(std::string_view name,
                                     std::chrono::milliseconds timeout,
                                     std::string& err,
                                     std::chrono::milliseconds io_timeout = {});

} // namespace coastd::client

#endif // COAST_COASTD_CLIENT_H
