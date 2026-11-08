#!/usr/bin/env python3
"""NSE vs 1-Hoyer dendrite noise gate: compare the trained arms (run from the worktree root).

Arms (work/<dir>): raw mnist (mnist / mnist_hoyer), mnist and chairs3d behind the retina
with k = 0.8 (<ds>_retina_k08 / <ds>_retina_k08_hoyer). Hoyer window 0.4/0.4, NSE 0.8/0.2.
"""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, "explore/nse_alt")
from ngm2_math import nse, sigmoid          # noqa: E402
from build_dnse import hoyer_noise, sep      # noqa: E402
from build_msm import mnist_labels, nmi      # noqa: E402

W = "explore/nse_alt/work"
STEPS = [1, 500, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000]
CONDS = ["real", "perm", "unif"]
THR = 0.3
MEAS = {"nse": (nse, 0.8, 0.2), "hoyer": (hoyer_noise, 0.4, 0.4)}
PAIRS = {"raw mnist": ("mnist", "mnist_hoyer", False),
         "retina mnist": ("mnist_retina_k08", "mnist_retina_k08_hoyer", True),
         "retina chairs3d": ("chairs3d_retina_k08", "chairs3d_retina_k08_hoyer", True)}


def L(d, c, s, k="out"):
    return np.load(f"{W}/{d}/resp/{c}_{s}.npz")[k].astype(np.float32)


def gate(m, a, start, width, steep):
    return 1 - sigmoid((m - start) / width, steep, 0.5) * (1 - sigmoid(a, 0.5, 0.5))


def gates(d, retina, meas):
    fn, st, wd = MEAS[meas]
    out = {}
    for c in CONDS:
        chans = [L(d, c, 500000, "ret_on"), L(d, c, 500000, "ret_off")] if retina else [L(d, c, 500000, "inp")]
        m = [fn(v) for v in chans]
        a = [v.mean(1) for v in chans]
        resp = gate(np.min(m, 0), np.min(a, 0), st, wd, 0.25)
        learn = [gate(mi, ai, st, wd, 0.5) for mi, ai in zip(m, a)]
        out[c] = {"resp": resp, "learn": learn, "meas": m}
    return out


def connectivity(d, step=500000):
    """Per dendrite: fraction of each input channel's inputs with >= 1 connected synapse."""
    f = h5py.File(f"{W}/{d}/ck/ck_{step}.h5")
    g = next(k for k in f["entities"] if "neurons" in f[f"entities/{k}"])
    rows, maps = [], None
    for nk in f[f"entities/{g}/neurons"]:
        for dk, dd in f[f"entities/{g}/neurons/{nk}/dendrites"].items():
            inc = dd["input_inc"][()].astype(np.int64) & 1
            idx = np.concatenate([[0], np.cumsum(inc)[:-1]])
            n_in = int(inc.sum())
            con = np.zeros(n_in, bool)
            con[idx[dd["permanence"][()] >= THR]] = True
            maps = con.astype(np.int64) if maps is None else maps + con
            rows.append((int(nk), int(dk), con))
    return rows, maps


def report(name, nse_dir, hoy_dir, retina):
    res = {}
    print(f"\n===== {name} =====")
    for meas, d in (("nse", nse_dir), ("hoyer", hoy_dir)):
        r = res[meas] = {}
        print(f"--- {meas:5s} ({d})")
        print(f"  {'step':>7s}  peak real/perm/unif     sep r-p  sep r-u  used")
        for s in STEPS:
            O = {c: L(d, c, s) for c in CONDS}
            pk = {c: O[c].max(1) for c in CONDS}
            used = len(np.unique(O["real"].argmax(1)))
            sp, su = sep(pk["real"], pk["perm"]), sep(pk["real"], pk["unif"])
            r[s] = dict(peak={c: float(np.median(pk[c])) for c in CONDS}, sep_rp=sp, sep_ru=su, used=used)
            print(f"  {s:7d}  {np.median(pk['real']):.3f} {np.median(pk['perm']):.3f} {np.median(pk['unif']):.3f}"
                  f"      {sp:+.2f}    {su:+.2f}    {used:2d}")
        G = gates(d, retina, meas)
        for c in CONDS:
            lr = " ".join(f"{np.median(x):.3f}" for x in G[c]["learn"])
            ms = " ".join(f"{np.median(x):.3f}" for x in G[c]["meas"])
            print(f"  gate@500k {c:4s}: measure(ch) {ms}  resp {np.median(G[c]['resp']):.3f}  learn(ch) {lr}")
        print(f"  gate sep real-perm {sep(G['real']['resp'], G['perm']['resp']):+.2f}"
              f"  real-unif {sep(G['real']['resp'], G['unif']['resp']):+.2f}")
        rows, maps = connectivity(d)
        nch = 2 if retina else 1
        n = len(rows[0][2]) // nch
        fr = np.array([[con[i * n:(i + 1) * n].mean() for i in range(nch)] for _, _, con in rows])
        print(f"  connected fraction per dendrite (ch): median {np.median(fr, 0).round(3)}  max {fr.max(0).round(3)}"
              f"  min {fr.min(0).round(3)}  (n_dend {len(rows)})")
        if retina:
            on, off = L(d, "real", 500000, "ret_on").mean(0), L(d, "real", 500000, "ret_off").mean(0)
            print(f"  channel-order check: corr(conn ON-part, mean ON) {np.corrcoef(maps[:n], on)[0, 1]:+.2f}"
                  f"  corr(conn OFF-part, mean OFF) {np.corrcoef(maps[n:], off)[0, 1]:+.2f}"
                  f"  crossed: {np.corrcoef(maps[:n], off)[0, 1]:+.2f} {np.corrcoef(maps[n:], on)[0, 1]:+.2f}")
        if "mnist" in d:
            lab = mnist_labels(L(d, "real", 500000, "inp"))
            J, score = nmi(L(d, "real", 500000).argmax(1), lab, 50, 10)
            print(f"  winners@500k: used {int((J.sum(0) > 0).sum())}  NMI(winner;digit) {score:.3f}")
            r["nmi"] = score
        r["conn_median"] = np.median(fr, 0).tolist()
    return res


if __name__ == "__main__":
    out = {k: report(k, *v) for k, v in PAIRS.items()}
    json.dump(out, open(f"{W}/hoyer_cmp.json", "w"), default=float)
