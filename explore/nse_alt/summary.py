#!/usr/bin/env python3
"""Numbers quoted in findings.html, all from work/<ds>/resp (pre-inhibition unless stated)."""
import numpy as np

import measures as M
from ngm2_math import soft_wta, soft_wta_factor

DS = ["mnist", "emnist", "kmnist", "cifar10", "dsprites", "chairs3d"]
L = lambda ds, c, s: np.load(f"work/{ds}/resp/{c}_{s}.npz")["out"].astype(np.float64)
sep = lambda a, b: 2 * M.auc(a, b) - 1


def spearman(a, b):
    return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]


print(f"{'ds':9s} {'NSE@1':>7s} {'NSE@500k':>8s} {'f@500k':>7s} {'rho(nse,cv)':>11s} | "
      f"sep vs perm: nse  gini  max | sep vs unif: nse  gini  max | sep vs step1: nse  max")
for ds in DS:
    r1, r = L(ds, "real", 1), L(ds, "real", 500000)
    p, u = L(ds, "perm", 500000), L(ds, "unif", 500000)
    m = {k: M.compute_all(v) for k, v in (("r1", r1), ("r", r), ("p", p), ("u", u))}
    V = np.concatenate([L(ds, c, s) for c in ("real", "perm", "unif") for s in (1, 5000, 50000, 500000)])
    mv = M.compute_all(V)
    f = float(np.median(soft_wta_factor(np.median(m["r"]["nse"]))))
    print(f"{ds:9s} {np.median(m['r1']['nse']):7.4f} {np.median(m['r']['nse']):8.4f} {f:7.3f} "
          f"{spearman(mv['nse'], mv['cv']):11.4f} | "
          f"{sep(m['r']['nse'], m['p']['nse']):+.2f} {sep(m['r']['gini'], m['p']['gini']):+.2f} "
          f"{sep(r.max(1), p.max(1)):+.2f} | "
          f"{sep(m['r']['nse'], m['u']['nse']):+.2f} {sep(m['r']['gini'], m['u']['gini']):+.2f} "
          f"{sep(r.max(1), u.max(1)):+.2f} | {sep(m['r']['nse'], m['r1']['nse']):+.2f} {sep(r.max(1), r1.max(1)):+.2f}")
print("post-inhibition NSE p10/p50/p90 at 500k (real):")
for ds in DS:
    print(f"  {ds:9s}", np.percentile(M.compute_all(soft_wta(L(ds, 'real', 500000)))["nse"], [10, 50, 90]).round(3))

# --- around the drop in the noise response, and the long runs (mnist, kmnist to 5M) ---
import h5py  # noqa: E402

PERM_THRESHOLD = 0.3   # configs/mnist.json ngm2.dendrites.default.permanence_threshold


def structure(ds, s):
    """(segments, connected synapses) summed over the group's dendrites."""
    f, segs, conn = h5py.File(f"work/{ds}/ck/ck_{s}.h5"), 0, 0
    for nk in f["entities/1/neurons"]:
        for d in f[f"entities/1/neurons/{nk}/dendrites"].values():
            segs += len(np.unique(d["segment_idx"][()]))
            conn += int((d["permanence"][()] >= PERM_THRESHOLD).sum())
    return segs, conn


print("\nreal vs perm (pre): median max real/perm, sep NSE, sep max")
for ds in ("mnist", "emnist", "kmnist"):
    for s in (100000, 200000, 300000, 400000, 500000):
        r, p = L(ds, "real", s), L(ds, "perm", s)
        print(f"  {ds:7s} {s:7d} {np.median(r.max(1)):.3f} {np.median(p.max(1)):.3f} "
              f"{sep(M.nse(r), M.nse(p)):+.2f} {sep(r.max(1), p.max(1)):+.2f}")
print("\nlong runs: median NSE real, median max real/perm/unif, sep NSE vs perm, segments, connected")
for ds in ("mnist", "kmnist"):
    for s in (500000, 1000000, 2000000, 5000000):
        V = {c: L(ds, c, s) for c in ("real", "perm", "unif")}
        print(f"  {ds:7s} {s:7d} {np.median(M.nse(V['real'])):.4f} "
              + " ".join(f"{np.median(v.max(1)):.3f}" for v in V.values())
              + f" {sep(M.nse(V['real']), M.nse(V['perm'])):+.2f} {structure(ds, s)}")
