#!/usr/bin/env python3
"""
analyze_ngm3_comparison.py

Side-by-side analysis of an ngm2 vs ngm3 comparison run produced by
scripts/ngm2_ngm3_compare.sh. Reads {ngm2,ngm3}/state_final.h5,
{ngm2,ngm3}/buffers.h5, and {ngm2,ngm3}/snapshots/*.h5 from a
comparison directory and emits five plots:

  - activity_histogram.png      — per-neuron activity over last 10k steps
  - rep_count_timeline.png      — total distinct seg_ids across checkpoints
  - permanence_hist_100k.png    — permanence distribution at ~100k steps
  - permanence_hist_final.png   — permanence distribution at end of run
  - specialization_heatmap.png  — per-class average activity heatmap
  - representation_samples.png  — learned permanence patterns for top-N neurons

mnist labels are reconstructed from cur_idx (mnist_input increments cur_idx
deterministically; change_interval=1 ⇒ image i at step i mod 60000).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import struct
import sys
from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# State file readers
# ---------------------------------------------------------------------------

@dataclass
class GroupState:
    """Common abstraction over ngm2 / ngm3 final state."""
    model: str            # "ngm2" or "ngm3"
    neuron_count: int
    # all permanence values flattened across the whole group
    permanences: np.ndarray
    # distinct seg_id count totalled across every dendrite of every neuron
    total_rep_count: int
    # per-neuron, per-dendrite: chosen permanence row of length input_size
    # (for the leaf with the most prominent representation in that dendrite)
    rep_samples: dict     # {neuron_idx: list[(label, permanence_row)]}


def _find_group_entity(h5: h5py.File) -> tuple[str, str]:
    """Return (entity_id, model) for the neuron group entity in this state."""
    for eid in h5["entities"].keys():
        g = h5["entities"][eid]
        if "neurons" in g:
            return eid, "ngm2"
        if "proximal" in g:
            return eid, "ngm3"
    raise RuntimeError("no neuron group entity found in state HDF5")


def read_ngm2_state(path: str) -> GroupState:
    with h5py.File(path, "r") as f:
        eid, _ = _find_group_entity(f)
        g = f["entities"][eid]
        neurons = g["neurons"]
        n_cnt = int(g["neuron_count"][()])
        all_perm = []
        total_seg = 0
        rep_samples: dict = {}
        for nid in sorted(neurons.keys(), key=int):
            nidx = int(nid)
            dendrites = neurons[nid]["dendrites"]
            best_row = None
            for did in sorted(dendrites.keys(), key=int):
                d = dendrites[did]
                perm = d["permanence"][:]
                seg = d["segment_idx"][:]
                all_perm.append(perm)
                total_seg += int(len(np.unique(seg)))
                if best_row is None:
                    best_row = perm.copy()  # first dendrite
            if best_row is not None:
                rep_samples[nidx] = [("dendrite 0", best_row)]
        return GroupState(
            model="ngm2",
            neuron_count=n_cnt,
            permanences=np.concatenate(all_perm) if all_perm else np.zeros(0),
            total_rep_count=total_seg,
            rep_samples=rep_samples,
        )


def read_ngm3_state(path: str) -> GroupState:
    with h5py.File(path, "r") as f:
        eid, _ = _find_group_entity(f)
        g = f["entities"][eid]
        n_cnt   = int(g["neuron_cnt"][()])
        a_cnt   = int(g["apical_cnt"][()])
        p_cnt   = int(g["proximal_cnt"][()])
        depth   = int(g["max_branch_level"][()])
        rep_cnt = 1 << depth
        max_seg = (1 << (depth + 1)) - 1
        all_perm_parts = []
        total_seg = 0
        rep_samples: dict = {}

        for bank, bcnt in (("proximal", p_cnt), ("apical", a_cnt)):
            if bcnt == 0:
                continue
            perm = g[bank]["permanences"][:]
            sids = g[bank]["segment_ids"][:]
            input_size = perm.size // (n_cnt * bcnt * rep_cnt)
            perm = perm.reshape(n_cnt, bcnt, rep_cnt, input_size)
            sids = sids.reshape(n_cnt, bcnt, rep_cnt, input_size)
            all_perm_parts.append(perm.reshape(-1))
            # distinct seg_ids per dendrite (across reps × syn_idx)
            for n in range(n_cnt):
                for d in range(bcnt):
                    distinct = np.unique(sids[n, d])
                    distinct = distinct[(distinct >= 1) & (distinct <= max_seg)]
                    total_seg += int(distinct.size)
            # representation sample: dendrite 0 of this bank, leaf at rep 0
            # (root segment if unbranched; left-most leaf otherwise)
            if bank == "proximal":
                for n in range(n_cnt):
                    rep_samples.setdefault(n, []).append(
                        (f"proximal[0] rep 0", perm[n, 0, 0].copy())
                    )

        return GroupState(
            model="ngm3",
            neuron_count=n_cnt,
            permanences=(np.concatenate(all_perm_parts)
                         if all_perm_parts else np.zeros(0)),
            total_rep_count=total_seg,
            rep_samples=rep_samples,
        )


def read_state(path: str) -> GroupState:
    with h5py.File(path, "r") as f:
        _, model = _find_group_entity(f)
    return read_ngm2_state(path) if model == "ngm2" else read_ngm3_state(path)


# ---------------------------------------------------------------------------
# Buffer log + mnist label sequence
# ---------------------------------------------------------------------------

def read_activity(buf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (step_numbers, neuron_activity[step, neuron]) from the
    group's output buffer (last buffer in the topology)."""
    with h5py.File(buf_path, "r") as f:
        names = [s.decode() for s in f["metadata/topology/buffer_names"][:]]
        # The group output is the buffer named like a neuron group
        out_idx = None
        for i, n in enumerate(names):
            if "neuron_group" in n:
                out_idx = i
                break
        if out_idx is None:
            out_idx = len(names) - 1
        steps = f["steps/step_numbers"][:]
        data = f[f"steps/buffers/{out_idx}/data"][:]
    return steps, data


def read_mnist_labels(label_file: str) -> np.ndarray:
    """Read mnist idx1 label file → uint8[N]."""
    with open(label_file, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801, f"bad magic {magic:#x}"
        return np.frombuffer(f.read(n), dtype=np.uint8)


def label_sequence(steps: np.ndarray,
                   label_file: str,
                   change_interval: int = 1) -> np.ndarray:
    """For each entry in `steps`, return the mnist label that was active.
    Sequential cur_idx + mod wrap with change_interval=1 means
    label[step % N_total]."""
    labels = read_mnist_labels(label_file)
    n = labels.size
    if change_interval <= 1:
        return labels[steps.astype(np.int64) % n]
    return labels[(steps.astype(np.int64) // change_interval) % n]


# ---------------------------------------------------------------------------
# Checkpoint discovery for rep-count timeline
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"_(\d+)\.h5$")


def discover_checkpoints(snapshot_dir: str) -> list[tuple[int, str]]:
    """Return [(step, path), ...] sorted by step. Snapshots are named
    <config>_<date>_<time>_<step>.h5 by run_headless's checkpointer."""
    out = []
    for path in glob.glob(os.path.join(snapshot_dir, "*.h5")):
        m = _CKPT_RE.search(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    out.sort()
    return out


def total_rep_count(state_path: str) -> int:
    return read_state(state_path).total_rep_count


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def fig_activity_histogram(out_dir: str,
                           ngm2_act: np.ndarray,
                           ngm3_act: np.ndarray,
                           window: int = 10000) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (label, a) in zip(axes, [("ngm2", ngm2_act), ("ngm3", ngm3_act)]):
        last = a[-window:, :].ravel() if a.shape[0] >= window else a.ravel()
        ax.hist(last, bins=60, range=(0, 1), color="steelblue", alpha=0.85)
        ax.set_title(f"{label} — activity hist (last {window} steps)")
        ax.set_xlabel("activity")
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "activity_histogram.png"), dpi=110)
    plt.close(fig)


def fig_rep_count_timeline(out_dir: str,
                           ngm2_ts: list[tuple[int, int]],
                           ngm3_ts: list[tuple[int, int]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, ts, color in [("ngm2", ngm2_ts, "tab:blue"),
                              ("ngm3", ngm3_ts, "tab:orange")]:
        if ts:
            xs, ys = zip(*ts)
            ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("total distinct seg_id count")
    ax.set_title("Representation growth (total distinct seg_ids over time)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rep_count_timeline.png"), dpi=110)
    plt.close(fig)


def fig_permanence_histogram(out_dir: str,
                             tag: str,
                             ngm2_perm: np.ndarray,
                             ngm3_perm: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (label, p) in zip(axes, [("ngm2", ngm2_perm), ("ngm3", ngm3_perm)]):
        if p.size:
            ax.hist(p, bins=60, range=(0, 1), color="darkorange", alpha=0.85)
        ax.set_title(f"{label} — permanence hist ({tag})")
        ax.set_xlabel("permanence")
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"permanence_hist_{tag}.png"), dpi=110)
    plt.close(fig)


def fig_specialization_heatmap(out_dir: str,
                               ngm2_act: np.ndarray,
                               ngm3_act: np.ndarray,
                               ngm2_labels: np.ndarray,
                               ngm3_labels: np.ndarray,
                               window: int = 10000) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, a, lbl) in zip(
        axes,
        [("ngm2", ngm2_act, ngm2_labels), ("ngm3", ngm3_act, ngm3_labels)],
    ):
        w = min(window, a.shape[0])
        act = a[-w:]
        lab = lbl[-w:]
        n_neurons = act.shape[1]
        heat = np.zeros((10, n_neurons), dtype=np.float32)
        for c in range(10):
            mask = (lab == c)
            if mask.any():
                heat[c] = act[mask].mean(axis=0)
        im = ax.imshow(heat, aspect="auto", cmap="viridis")
        ax.set_title(f"{label} — class × neuron avg activity (last {w} steps)")
        ax.set_xlabel("neuron")
        ax.set_ylabel("mnist class")
        ax.set_yticks(range(10))
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "specialization_heatmap.png"), dpi=110)
    plt.close(fig)


def fig_representation_samples(out_dir: str,
                               ngm2_state: GroupState,
                               ngm3_state: GroupState,
                               selected_neurons: list[int],
                               img_shape: tuple[int, int] = (28, 28)) -> None:
    n_sel = len(selected_neurons)
    fig, axes = plt.subplots(2, n_sel, figsize=(2 * n_sel, 4.5))
    if n_sel == 1:
        axes = axes.reshape(2, 1)
    for col, n_idx in enumerate(selected_neurons):
        for row, (label, st) in enumerate([("ngm2", ngm2_state),
                                            ("ngm3", ngm3_state)]):
            ax = axes[row, col]
            samples = st.rep_samples.get(n_idx, [])
            if samples:
                _name, perm_row = samples[0]
                try:
                    img = perm_row[: img_shape[0] * img_shape[1]].reshape(img_shape)
                except ValueError:
                    img = np.zeros(img_shape, dtype=np.float32)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"n={n_idx}")
    fig.suptitle("Representation samples (selected neurons, dendrite 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "representation_samples.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def pick_selected_neurons(ngm2_act: np.ndarray,
                          ngm3_act: np.ndarray,
                          ngm2_labels: np.ndarray,
                          ngm3_labels: np.ndarray,
                          window: int = 10000,
                          k: int = 8) -> list[int]:
    """Pick neurons with the strongest class specialisation in ngm3,
    capped at min(k, n2, n3) so the figure stays comparable."""
    w = min(window, ngm3_act.shape[0])
    act = ngm3_act[-w:]
    lab = ngm3_labels[-w:]
    heat = np.zeros((10, act.shape[1]), dtype=np.float32)
    for c in range(10):
        mask = (lab == c)
        if mask.any():
            heat[c] = act[mask].mean(axis=0)
    # specialisation score = max class avg minus mean of other classes
    top_class = heat.max(axis=0)
    other_mean = (heat.sum(axis=0) - top_class) / 9
    score = top_class - other_mean
    n_avail = min(ngm2_act.shape[1], ngm3_act.shape[1])
    k = min(k, n_avail)
    idx = np.argsort(-score)[:k]
    return sorted(int(i) for i in idx if i < n_avail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("compare_dir", help="benchmarks/ngm3_comparison_<sha>/")
    ap.add_argument("--mnist-labels", default="./data/mnist/train-labels-idx1-ubyte")
    ap.add_argument("--window", type=int, default=10000,
                    help="trailing-window size for activity-based figures")
    args = ap.parse_args()

    ngm2_dir = os.path.join(args.compare_dir, "ngm2")
    ngm3_dir = os.path.join(args.compare_dir, "ngm3")
    if not os.path.isdir(ngm2_dir) or not os.path.isdir(ngm3_dir):
        print(f"error: expected {ngm2_dir} and {ngm3_dir} to exist", file=sys.stderr)
        sys.exit(1)

    out_dir = args.compare_dir
    print(f"[analyze] reading from: {args.compare_dir}")

    # --- final states ---------------------------------------------------
    ngm2_state = read_state(os.path.join(ngm2_dir, "state_final.h5"))
    ngm3_state = read_state(os.path.join(ngm3_dir, "state_final.h5"))
    print(f"[analyze] ngm2: {ngm2_state.neuron_count} neurons, "
          f"{ngm2_state.total_rep_count} total reps, "
          f"{ngm2_state.permanences.size} permanence values")
    print(f"[analyze] ngm3: {ngm3_state.neuron_count} neurons, "
          f"{ngm3_state.total_rep_count} total reps, "
          f"{ngm3_state.permanences.size} permanence values")

    # --- buffer logs (activity time series) -----------------------------
    ngm2_steps, ngm2_act = read_activity(os.path.join(ngm2_dir, "buffers.h5"))
    ngm3_steps, ngm3_act = read_activity(os.path.join(ngm3_dir, "buffers.h5"))
    ngm2_labels = label_sequence(ngm2_steps, args.mnist_labels)
    ngm3_labels = label_sequence(ngm3_steps, args.mnist_labels)

    # --- 1: activity histogram -----------------------------------------
    fig_activity_histogram(out_dir, ngm2_act, ngm3_act, window=args.window)

    # --- 2: rep-count timeline -----------------------------------------
    ngm2_ts: list[tuple[int, int]] = []
    ngm3_ts: list[tuple[int, int]] = []
    for label, snap_dir, ts in [
        ("ngm2", os.path.join(ngm2_dir, "snapshots"), ngm2_ts),
        ("ngm3", os.path.join(ngm3_dir, "snapshots"), ngm3_ts),
    ]:
        for step, path in discover_checkpoints(snap_dir):
            try:
                ts.append((step, total_rep_count(path)))
            except Exception as e:
                print(f"  warn: failed to read {path}: {e}", file=sys.stderr)
    # Always include the final state
    if ngm2_steps.size:
        ngm2_ts.append((int(ngm2_steps[-1]) + 1, ngm2_state.total_rep_count))
    if ngm3_steps.size:
        ngm3_ts.append((int(ngm3_steps[-1]) + 1, ngm3_state.total_rep_count))
    ngm2_ts.sort(); ngm3_ts.sort()
    fig_rep_count_timeline(out_dir, ngm2_ts, ngm3_ts)

    # --- 3: permanence histograms at ~100k and final --------------------
    # ~100k: pick the checkpoint whose step is closest to 100000
    def pick_near(snap_dir: str, target: int) -> str | None:
        cks = discover_checkpoints(snap_dir)
        if not cks: return None
        best = min(cks, key=lambda x: abs(x[0] - target))
        return best[1]
    ngm2_100k = pick_near(os.path.join(ngm2_dir, "snapshots"), 100000)
    ngm3_100k = pick_near(os.path.join(ngm3_dir, "snapshots"), 100000)
    ngm2_p100 = read_state(ngm2_100k).permanences if ngm2_100k else np.zeros(0)
    ngm3_p100 = read_state(ngm3_100k).permanences if ngm3_100k else np.zeros(0)
    fig_permanence_histogram(out_dir, "100k", ngm2_p100, ngm3_p100)
    fig_permanence_histogram(out_dir, "final",
                              ngm2_state.permanences, ngm3_state.permanences)

    # --- 4: specialization heatmap -------------------------------------
    fig_specialization_heatmap(out_dir, ngm2_act, ngm3_act,
                                ngm2_labels, ngm3_labels, window=args.window)

    # --- 5: representation samples -------------------------------------
    selected = pick_selected_neurons(ngm2_act, ngm3_act,
                                      ngm2_labels, ngm3_labels,
                                      window=args.window, k=8)
    fig_representation_samples(out_dir, ngm2_state, ngm3_state, selected)

    print(f"[analyze] figures written to: {out_dir}")
    for name in ("activity_histogram.png", "rep_count_timeline.png",
                 "permanence_hist_100k.png", "permanence_hist_final.png",
                 "specialization_heatmap.png", "representation_samples.png"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
