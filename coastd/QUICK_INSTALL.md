# `coastd` quick install — checklist

The bare sequence of steps, nothing else. Every step links to the section of
[`INSTALL.md`](INSTALL.md) that explains *why* it exists — read that one first if
this is your first install, or whenever a step here fails.

Linux/systemd only. On macOS the daemon runs only when started by hand
([Appendix A](INSTALL.md#appendix-a--macos)); clients are portable.

Run everything as the dedicated daemon user.

---

## 1. Prerequisites — [§1](INSTALL.md#1-prerequisites)

- [ ] Install the toolchain (C++23 compiler, CMake ≥ 3.24, git, TBB, HDF5, ninja/make).
      No SDL2/OpenGL/GPU/X server needed.
      ```sh
      sudo apt install build-essential cmake git ninja-build libtbb-dev libhdf5-dev   # Debian/Ubuntu
      sudo pacman -S base-devel cmake git ninja onetbb hdf5                           # Arch
      ```
- [ ] Confirm the machine can reach the git remote (submodules + FetchContent run
      at *runtime*, once per SHA) — [§1.2](INSTALL.md#12-network-access).
- [ ] Create the unprivileged daemon user and switch to it — [§1.3](INSTALL.md#13-a-dedicated-user-account).
      ```sh
      sudo useradd --create-home --shell /bin/bash coast
      sudo -iu coast
      ```

## 2. Source and datasets — [§2](INSTALL.md#2-get-the-source)

- [ ] Clone with submodules.
      ```sh
      git clone --recurse-submodules <your-coast-remote> ~/src/coast
      cd ~/src/coast
      ```
- [ ] Fetch the datasets once (or point at an existing shared copy) — [§2.1](INSTALL.md#21-datasets).
      ```sh
      ./scripts/download_datasets.sh
      ```
- [ ] Note the absolute dataset directory — you set it as `dataset_base` in step 6.

## 3. Build the daemon — [§3](INSTALL.md#3-build-coastd)

- [ ] Configure.
      ```sh
      cmake -B build-coastd \
            -DCMAKE_BUILD_TYPE=Release \
            -DCOAST_WITH_GUI=OFF \
            -DCOAST_GPU_BACKEND=none \
            -DCOAST_WITH_CONSOLE=OFF
      ```
- [ ] Build the `coastd` target only.
      ```sh
      cmake --build build-coastd -j --target coastd
      ```
- [ ] Verify it runs — note the binary is one level down, in `build-coastd/coastd/`.
      ```sh
      ./build-coastd/coastd/coastd --version     # -> coastd 0.1.0-dev
      ```
- [ ] Copy the binary out of the build tree — [§3.3](INSTALL.md#33-install-the-binary-somewhere-stable).
      ```sh
      mkdir -p ~/.local/bin && install -m 755 build-coastd/coastd/coastd ~/.local/bin/coastd
      ```

## 4. Test before installing — [§4](INSTALL.md#4-test-the-build-before-installing-it)

- [ ] Configure + build the test tree with the **same** flags plus `-DBUILD_TESTING=ON`.
      ```sh
      cmake -B build-tests -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
            -DCOAST_WITH_GUI=OFF -DCOAST_GPU_BACKEND=none -DCOAST_WITH_CONSOLE=OFF
      cmake --build build-tests -j
      ```
- [ ] Run the daemon suite — expect **24/24**.
      ```sh
      cd build-tests && ctest --output-on-failure -R '^unit\.coastd\.'
      ```
- [ ] *(optional, slow)* Fault/latency matrix: `ctest --output-on-failure -R '^fault\.coastd\.'`.
- [ ] *(optional, slow)* Real end-to-end bit-identity gate — [§4.3](INSTALL.md#43-optional-the-real-end-to-end-gate).
      Needs `h5diff`, `python3` and the datasets; exits *Skipped* if any is missing.
      ```sh
      cmake --build build-tests -j --target coast
      ctest --output-on-failure -R '^gate\.coastd\.real_coast$'
      ```

## 5. Prepare the roots — [§5](INSTALL.md#5-choose-the-layout-and-prepare-the-roots)

- [ ] Create the data root (everything under it is created by the daemon itself).
      ```sh
      mkdir -p ~/.local/share/coastd
      ```
- [ ] Decide whether `runs_root` (large) belongs on a different volume than
      `state_root` (tiny, back this up) — [§5.1](INSTALL.md#51-data-root).
- [ ] Decide ref-freshness policy: leave fetching on (default) or run `--no-fetch`
      with your own fetch discipline — [§5.2](INSTALL.md#52-repo-root-and-ref-freshness).

## 6. Write the config — [§6](INSTALL.md#6-configure-the-daemon)

- [ ] Create `~/.local/share/coastd/coastd.json`. Minimum viable starting point:
      ```json
      {
        "slots": 2,
        "dataset_base": "/home/coast/src/coast/data",
        "checkpoint_interval": 500000,
        "build_jobs": 8,
        "build_cmake_args": [
          "-DCOAST_WITH_GUI=OFF",
          "-DCOAST_GPU_BACKEND=none",
          "-DCOAST_WITH_CONSOLE=OFF"
        ],
        "term_grace_seconds": 30
      }
      ```
- [ ] **Check `build_cmake_args` matches step 3's flags.** The host build's flags do
      not propagate to the per-SHA builds; a missing `-DCOAST_WITH_GUI=OFF` makes
      every job's build *fail* at configure on a server without SDL2 — [§6.2](INSTALL.md#62-a-starting-config).
- [ ] Set `dataset_base` (or commit to absolute dataset paths in every config you
      submit) — [§2.1](INSTALL.md#21-datasets).
- [ ] Raise `term_grace_seconds` above the time your largest model needs to write a
      final checkpoint (default 10 s).
- [ ] Confirm TCP 51900 / UDP 51901 are free, or set `tcp_port`/`udp_port` — [§6.3](INSTALL.md#63-ports).

## 7. Install the service — [§7](INSTALL.md#7-install-as-a-systemd-user-service)

- [ ] Generate the unit from the installed binary.
      ```sh
      ~/.local/bin/coastd --install-service \
          --data-root ~/.local/share/coastd \
          --repo ~/src/coast
      ```
- [ ] Enable and start it.
      ```sh
      systemctl --user daemon-reload
      systemctl --user enable --now coastd
      ```
- [ ] Enable lingering — **without this the daemon dies when you log out**.
      ```sh
      sudo loginctl enable-linger "$USER"
      ```
- [ ] Do **not** edit `KillMode` in the unit — [§7.2](INSTALL.md#72-do-not-override-killmode).
- [ ] Check it booted.
      ```sh
      systemctl --user status coastd
      journalctl --user -u coastd -f     # expect a startup line, then "booted: N active job(s)"
      ```

## 8. Verify end to end — [§8](INSTALL.md#8-verify-end-to-end)

- [ ] Talk to it.
      ```sh
      printf '{"cmd":"machine_info"}\n' | nc -q1 localhost 51900
      ```
      Check `data_link_ok:true` if you set `dataset_base`.
- [ ] Submit a real job (config travels as **content**, never a path).
      ```sh
      jq -cn --arg ref main --rawfile cfg ./configs/mnist.json \
         '{cmd:"submit", ref:$ref, config_text:$cfg, config_name:"mnist.json",
           steps_target:1000000, label:"smoke", message:"install verification"}' \
        | nc -q1 localhost 51900
      ```
      Always set `steps_target` — `0` means *unbounded*.
- [ ] Watch it through `queued → resolving → building → ready → running → done`.
      ```sh
      printf '{"cmd":"list"}\n' | nc -q1 localhost 51900
      tail -f ~/.local/share/coastd/worktrees/<sha>/coastd-build.log
      tail -f ~/.local/share/coastd/runs/<job-id>/logs/run_0_0_0.log
      ```
- [ ] **Test recovery** — restart the daemon mid-run and confirm the job returns as
      `running` from its last checkpoint — [§8.4](INSTALL.md#84-test-recovery).
      ```sh
      systemctl --user restart coastd
      printf '{"cmd":"list"}\n' | nc -q1 localhost 51900
      ```
- [ ] Test cancellation.
      ```sh
      printf '{"cmd":"cancel","id":"<job-id>"}\n' | nc -q1 localhost 51900
      ```

## 9. Secure the port — [§9](INSTALL.md#9-security--read-this-before-exposing-the-port)

- [ ] **Anyone who can reach TCP 51900 has remote code execution as the daemon
      user. There is no authentication.** Do not skip this step.
- [ ] Firewall both ports to known clients (minimum acceptable step).
      ```sh
      sudo ufw allow from 10.0.0.0/24 to any port 51900 proto tcp
      sudo ufw allow from 10.0.0.0/24 to any port 51901 proto udp
      ```
- [ ] Or restrict to `lo` and have clients use `ssh -L 51900:localhost:51900 coast@server`.

## 10. Operational follow-up — [§10](INSTALL.md#10-operating-it)

- [ ] Set up external monitoring of free space on the `runs_root` volume — the
      daemon stops starting work below `min_free_disk_bytes` (2 GiB) but never
      deletes run directories itself.
- [ ] Decide and schedule a retention policy for `<runs_root>/<id>/` (job logs are
      compressed and eventually deleted on their own; run directories are not).
- [ ] Back up `<data-root>/state/` — that is the queue.
- [ ] Note the upgrade sequence — [§10.2](INSTALL.md#102-upgrading).
      ```sh
      cd ~/src/coast && git pull --recurse-submodules
      cmake --build build-coastd -j --target coastd
      install -m 755 build-coastd/coastd/coastd ~/.local/bin/coastd   # not cp: "Text file busy"
      systemctl --user restart coastd
      ```

---

Something failed? → [§11 Troubleshooting](INSTALL.md#11-troubleshooting).
