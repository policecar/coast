# Running coastd as a service

coastd is designed to run as a **resident per-user background service**. Two
deployment targets are provided; **Linux/systemd is the primary, supported one.**

## Linux — systemd user service (primary)

The quickest path is the built-in installer, which writes a unit pointing at the
*currently running* `coastd` binary and the given roots:

```sh
coastd --install-service \
    --data-root ~/.local/share/coastd \
    --repo ~/src/coast
```

It writes `~/.config/systemd/user/coastd.service` and prints the enable steps:

```sh
systemctl --user daemon-reload
systemctl --user enable --now coastd
sudo loginctl enable-linger "$USER"   # start at BOOT without an interactive login
```

Inspect / control:

```sh
systemctl --user status coastd
journalctl --user -u coastd -f        # follow the daemon log
systemctl --user restart coastd       # safe: R1 relaunches interrupted jobs
```

To install by hand instead, copy [`coastd.service`](coastd.service) and edit its
`ExecStart` line.

### Why a *user* service, and why linger

A **user** service inherits your git credentials (coastd builds git refs) and
environment. `loginctl enable-linger` lets that user service start at boot with
no interactive login — the headless deployment coastd targets.

### Recovery model (R1) — do not override `KillMode`

The unit keeps systemd's **stock `KillMode=control-group`**. On stop, restart, or
crash, systemd reaps the entire cgroup, so every child `coast` process dies with
the daemon. That is the whole recovery contract: **no live orphans, ever** — each
restart relaunches interrupted jobs from their last checkpoint, and two children
can never clobber the same output. Setting `KillMode=process`/`mixed` would leave
orphans and break this guarantee.

## macOS — not a deployment target

**The daemon runs on macOS, but only started by hand** (ADR-17 in
`docs/content/coastd/`, superseding ADR-16's macOS clause).

launchd does not reap a process group the way systemd's cgroup kill does, so the
"no live orphans" property R1 relies on is **not automatic** there. coastd's
boot-time **pid duplicate-guard** covers it, as it does on Linux outside systemd:
it identifies a surviving orphan by its start time, read on macOS from
`sysctl(KERN_PROC_PID)` (see
[`../include/daemon/child_proc.h`](../include/daemon/child_proc.h)), and kills it
before relaunching.

Every **client** is portable (pure TCP): the GUI `coastd` panel, the console
`coast::coastd` façade and `coastd_client`.

[`com.coast.coastd.plist`](com.coast.coastd.plist) is **unused and unvalidated**:
unattended operation under launchd is out of scope. It has never been run, and
start at boot, restart on failure and git credentials are unverified there.
