"""A client for coastd, the per-machine job daemon (`docs/content/coastd/protocol.md`).

The wire is JSON lines over TCP: one request object per line, one reply per line, every
reply `{"ok": true, ...}` or `{"ok": false, "error": "..."}`. Each call here opens its own
connection: the protocol has no request ids, so a reply that timed out on a kept-open
connection would be read as the answer to the next request.

Every failure raises DaemonError naming what went wrong; a malformed reply is never
returned or replaced by a default.

No request has a deadline. How long a reply takes depends on the daemon's load, its disk and
the NAS (`hash_inputs` reads every input), so a timeout would guess. Instead the connection
uses TCP keepalive: the daemon host's kernel answers the probes while coastd works, and on
an idle connection a host that goes down or is cut off fails the request after
`keepalive_idle_s` + `keepalive_count` x `keepalive_interval_s` of silence. The OS bounds
the two cases keepalive does not cover: a connect to a host already down (SYN retries, ~2
min on Linux) and a request sent but not yet acknowledged (retransmission, ~15 min). A daemon
that is alive but never answers is waited for; interrupt the run and resume it
(`runner.run_milestone1`).
"""

from __future__ import annotations

import json
import re
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import msgspec

from coast_eval.runs import RunDir

PROTO_VERSION = 1  # `proto_version` in coastd/include/daemon/protocol.h
DEFAULT_TCP_PORT = 51900  # `default_tcp_port` in coastd/include/daemon/net_defaults.h
DEFAULT_UDP_PORT = 51901  # `default_udp_port`, the discovery responder
DISCOVERY_PROBE = b"coastd-probe-v1"  # `discovery_probe` in protocol.h
Bucket = Literal["active", "done", "failed", "cancelled", "quarantine"]


class DaemonError(Exception):
    pass


class DaemonUnreachable(DaemonError):
    """No reply because the connection failed (refused, reset, timed out): the one error a
    caller may retry for an idempotent request such as `list`."""


class JobSummary(msgspec.Struct, frozen=True):
    """One entry of `list`, reduced to the fields this client reads; the others are ignored,
    as are fields a newer daemon adds without a version bump. Each declared field is required
    and must have its type. `bucket` is the authoritative lifecycle state."""

    id: str
    bucket: Bucket
    submitted_at: str  # ISO-8601 UTC, to the second: the newest job of a key wins
    failure_reason: str
    user_message: str  # the submit's `message`: the eval tags its jobs with it


class MachineInfo(msgspec.Struct, frozen=True):
    hostname: str  # the daemon's host
    runs_root: str  # a path on the daemon's host
    data_link_ok: bool  # each run dir gets `./data` linked to the daemon's dataset_base
    dataset_base: str  # that link's target, a path on the daemon's host ("" if unset)
    tcp_port: int  # the port the daemon listens on: with hostname, the daemon's identity


TOO_OLD = "daemon too old for multi-daemon evaluation (needs coastd with hash_inputs)"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class _Hashes(msgspec.Struct, frozen=True):
    sha256: dict[str, str]


class _Submitted(msgspec.Struct, frozen=True):
    id: str


class _Listed(msgspec.Struct, frozen=True):
    jobs: list[JobSummary]


class _Ports(msgspec.Struct, frozen=True):  # `<data-root>/coastd.ports`
    tcp_port: int


def _keepalive(s: socket.socket, idle_s: int, interval_s: int, count: int) -> None:
    """Turn on TCP keepalive with the given probe timing (Linux and macOS names)."""
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    idle = "TCP_KEEPIDLE" if hasattr(socket, "TCP_KEEPIDLE") else "TCP_KEEPALIVE"  # macOS
    for name, value in ((idle, idle_s), ("TCP_KEEPINTVL", interval_s), ("TCP_KEEPCNT", count)):
        opt = getattr(socket, name, None)
        if opt is None:
            raise DaemonError(f"this platform has no socket.{name}: cannot set TCP keepalive")
        s.setsockopt(socket.IPPROTO_TCP, opt, value)


@dataclass(frozen=True)
class Client:
    host: str
    port: int
    # Liveness of the daemon's host, not a deadline on the reply (module docstring).
    keepalive_idle_s: int = 10
    keepalive_interval_s: int = 5
    keepalive_count: int = 3

    @classmethod
    def from_ports_file(cls, path: Path, host: str = "127.0.0.1") -> Client:
        """Connect to the daemon that wrote `<data-root>/coastd.ports`."""
        try:
            port = msgspec.json.decode(path.read_bytes(), type=_Ports).tcp_port
        except msgspec.DecodeError as e:
            raise DaemonError(f"{path}: {e}") from e
        return cls(host, port)

    def request(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Send one request line, return the decoded `ok: true` reply."""
        line = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
        try:
            # no timeout: connect as long as the OS retries, then wait on liveness alone
            with socket.create_connection((self.host, self.port)) as s:
                _keepalive(s, self.keepalive_idle_s, self.keepalive_interval_s,
                           self.keepalive_count)  # fmt: skip
                s.sendall(line)
                raw = s.makefile("rb").readline()
        except OSError as e:
            raise DaemonUnreachable(f"coastd at {self.host}:{self.port}: {e}") from e
        what = f"coastd reply to {obj.get('cmd')!r}"
        if not raw:  # closed before replying at all: the daemon stopped or restarted
            raise DaemonUnreachable(f"{what}: connection closed before any reply")
        if not raw.endswith(b"\n"):
            raise DaemonError(f"{what}: connection closed before a complete line: {raw!r}")
        try:
            reply = json.loads(raw)
        except ValueError as e:
            raise DaemonError(f"{what}: not JSON: {raw!r}") from e
        if not isinstance(reply, dict) or not isinstance(
            cast(dict[str, Any], reply).get("ok"), bool
        ):
            raise DaemonError(f"{what}: not an envelope with a boolean 'ok': {raw!r}")
        reply = cast(dict[str, Any], reply)
        if not reply["ok"]:
            raise DaemonError(f"{what}: {reply.get('error')!r}")
        if reply.get("proto_version") != PROTO_VERSION:
            raise DaemonError(f"{what}: proto_version {reply.get('proto_version')!r}, "
                              f"this client speaks {PROTO_VERSION}")  # fmt: skip
        return reply

    def _typed[T](self, obj: dict[str, Any], kind: type[T]) -> T:
        try:
            return msgspec.convert(self.request(obj), kind)
        except msgspec.ValidationError as e:
            raise DaemonError(f"coastd reply to {obj['cmd']!r}: {e}") from e

    def submit(
        self,
        ref: str,
        config_text: str,
        steps: int,
        *,
        priority: int,
        load_state_from: str | None = None,
        message: str | None = None,
    ) -> str:
        """Queue a job; returns its id. With `load_state_from` (a done job's id) the job
        starts from that job's final state and runs `steps` more steps. `message` is stored
        with the job and listed back as `user_message`."""
        if steps < 1:
            raise ValueError(f"steps must be >= 1 (0 is unbounded on coastd), got {steps}")
        req = {"cmd": "submit", "ref": ref, "config_text": config_text,
               "steps_target": steps, "priority": priority}  # fmt: skip
        if load_state_from is not None:
            req["load_state_from"] = load_state_from
        if message is not None:
            req["message"] = message
        return self._typed(req, _Submitted).id

    def list_jobs(self) -> list[JobSummary]:
        return self._typed({"cmd": "list"}, _Listed).jobs

    def cancel(self, job_id: str) -> None:
        """Stop a job (SIGTERM, then SIGKILL after coastd's grace); raises if it is unknown or
        has already ended."""
        self.request({"cmd": "cancel", "id": job_id})

    def machine_info(self) -> MachineInfo:
        reply = self.request({"cmd": "machine_info"})
        if "dataset_base" not in reply:
            raise DaemonError(f"coastd at {self.host}:{self.port}: {TOO_OLD}")
        try:
            return msgspec.convert(reply, MachineInfo)
        except msgspec.ValidationError as e:
            raise DaemonError(f"coastd reply to 'machine_info': {e}") from e

    def hash_inputs(self, paths: Sequence[str]) -> dict[str, str]:
        """SHA256 (lower-case hex) of each path, relative to the daemon's dataset_base and
        read on the daemon's filesystem. The reply must name exactly the requested paths."""
        if not paths:
            raise ValueError("hash_inputs needs at least one path")
        try:
            got = self._typed({"cmd": "hash_inputs", "paths": list(paths)}, _Hashes).sha256
        except DaemonError as e:
            if "unknown command" in str(e):
                raise DaemonError(f"coastd at {self.host}:{self.port}: {TOO_OLD}") from e
            raise
        if got.keys() != set(paths):
            raise DaemonError(f"coastd reply to 'hash_inputs': hashes for {sorted(got)}, "
                              f"asked for {sorted(set(paths))}")  # fmt: skip
        bad = sorted(p for p, h in got.items() if not _SHA256_HEX.fullmatch(h))
        if bad:
            raise DaemonError(f"coastd reply to 'hash_inputs': not a SHA256 hex digest for {bad}")
        return got


@dataclass(frozen=True)
class DaemonSpec:
    """One `--daemon HOST[:PORT][=LOCAL_RUNS_ROOT]`. `local_runs_root` is where this host
    sees the daemon's runs_root, when the mount point differs; None means the same path."""

    host: str
    port: int
    local_runs_root: Path | None = None

    def client(self) -> Client:
        return Client(self.host, self.port)


def parse_daemon_spec(spec: str) -> DaemonSpec:
    """Parse HOST[:PORT][=LOCAL_RUNS_ROOT]; without a port, coastd's default. An IPv6 host
    is bracketed, with or without a port ([::1], [::1]:51900)."""
    addr, sep, root = spec.partition("=")
    m = re.fullmatch(r"\[([0-9A-Fa-f:.]+)\](?::(\d+))?|([^:\[\]\s]+)(?::(\d+))?", addr)
    if m is None:
        raise ValueError(f"daemon {spec!r} is not HOST[:PORT][=LOCAL_RUNS_ROOT]")
    host, port = (m[1], m[2]) if m[1] else (m[3], m[4])
    port = port or str(DEFAULT_TCP_PORT)
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"daemon {spec!r}: port {port} is not in 1..65535")
    if sep and not root:
        raise ValueError(f"daemon {spec!r}: empty LOCAL_RUNS_ROOT after '='")
    return DaemonSpec(host, int(port), Path(root) if sep else None)


class _Discovered(msgspec.Struct, frozen=True):  # a discovery reply, reduced
    hostname: str
    tcp_port: int
    proto_version: int


@dataclass(frozen=True, order=True)
class Discovered:
    """A daemon that answered a discovery probe; `address` is where its reply came from."""

    hostname: str
    address: str
    tcp_port: int
    proto_version: int

    @property
    def spec(self) -> str:
        """The `--daemon` argument that names this daemon."""
        port = "" if self.tcp_port == DEFAULT_TCP_PORT else f":{self.tcp_port}"
        return f"{self.address}{port}"


def discover(
    timeout_s: float = 2.0, port: int = DEFAULT_UDP_PORT, broadcast: str = "255.255.255.255"
) -> list[Discovered]:
    """Broadcast one probe on the local network and collect the daemons that answer within
    `timeout_s`, sorted by hostname. Best effort: broadcasts do not cross routers or VPN
    tunnels, and a datagram that is not a coastd reply is skipped. `broadcast` is the
    probe's destination (a test probes 127.0.0.1)."""
    found: set[Discovered] = set()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(DISCOVERY_PROBE, (broadcast, port))
        deadline = time.monotonic() + timeout_s
        while (left := deadline - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                data, (addr, _) = s.recvfrom(65536)
            except TimeoutError:
                break
            try:
                r = msgspec.json.decode(data, type=_Discovered)
            except (msgspec.DecodeError, msgspec.ValidationError):
                continue
            found.add(Discovered(r.hostname, addr, r.tcp_port, r.proto_version))
    return sorted(found)


def verify_daemon_inputs(client: Client, expected: Mapping[str, str]) -> None:
    """Raise unless the daemon reports, for every path (relative to its dataset_base), the
    expected SHA256. Names every path that differs, not only the first."""
    got = client.hash_inputs(list(expected))
    bad = [f"{p}: daemon {got[p]}, expected {sha}" for p, sha in expected.items() if got[p] != sha]
    if bad:
        raise DaemonError(f"coastd at {client.host}:{client.port}: inputs differ from the "
                          "expected hashes:\n  " + "\n  ".join(bad))  # fmt: skip


def run_dir(job_id: str, runs_root: Path) -> RunDir:
    """A job's run directory, `<runs_root>/<id>` (`docs/content/coastd/persistence.md`)."""
    if not job_id or Path(job_id).name != job_id or job_id in (".", ".."):
        raise DaemonError(f"job id {job_id!r} is not a single path component")
    return RunDir(runs_root / job_id)
