#!/usr/bin/env python3
"""Per-gate window fit for the 1 - Hoyer dendrite noise gate (run from the worktree root).

Fit data: hoyer_map family P (simulated post-inhibition group outputs), N in {50..400},
restricted to floors bg <= a / 10 (max_softmax leaves practically no floor).
Targets on the full attenuation 1 - g * (1 - s(avg)) (user, 2026-10-26):
  response: signal >= 0.9, middle in [0.35, 0.65], noise <= 0.1
  learning: signal >= 0.9, middle in [0.15, 0.35], noise <= 0.1
Checks (not fitted): real 50-neuron outputs after max_softmax (family R), retina inputs (family D).
Fits each gate alone and both with one shared (start, width), as the C++ gate has.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "explore/nse_alt")
import hoyer_map as hm   # noqa: E402

TARGET = {"resp": {"signal": (0.9, 1.0), "middle": (0.35, 0.65), "noise": (0.0, 0.1)},
          "learn": {"signal": (0.9, 1.0), "middle": (0.15, 0.35), "noise": (0.0, 0.1)}}
CLASSES = ["signal", "middle", "noise"]


def labelled(P):
    # noise only at a <= 0.3: bright flat inputs bypass the gate via the avg term by design,
    # and real flat outputs after max_softmax peak at 0.06-0.30
    rows = [r for r in P if r["cls"] and r["bg"] <= r["a"] / 10 and not (r["cls"] == "noise" and r["a"] > 0.3)]
    X = np.concatenate([r["h"] for r in rows])
    A = np.concatenate([r["avg"] for r in rows])
    C = np.concatenate([[r["cls"]] * len(r["h"]) for r in rows])
    Nn = np.concatenate([[r["N"]] * len(r["h"]) for r in rows])
    return X, A, C, Nn


def gate_loss(X, A, C, Nn, st, wd, gate):
    att = hm.atten(X, A, st, wd, hm.STEEP[gate])
    lo = np.array([TARGET[gate][c][0] for c in C], np.float32)
    hi = np.array([TARGET[gate][c][1] for c in C], np.float32)
    v = np.maximum(lo - att, 0) + np.maximum(att - hi, 0)
    return float(np.mean([np.mean(v[(C == c) & (Nn == n)] ** 2) for c in CLASSES for n in hm.NS]))


def search(lossf):
    best = (np.inf, None)
    for st in np.arange(-1.0, 1.0, 0.02):
        for wd in np.arange(0.05, 2.0, 0.02):
            L = lossf(st, wd)
            if L < best[0]:
                best = (L, (st, wd))
    s0, w0 = best[1]
    for st in np.arange(s0 - 0.03, s0 + 0.03, 0.005):
        for wd in np.arange(max(0.02, w0 - 0.03), w0 + 0.03, 0.005):
            L = lossf(st, wd)
            if L < best[0]:
                best = (L, (round(float(st), 3), round(float(wd), 3)))
    return best[1], best[0]


def in_target(X, A, C, Nn, st, wd, gate):
    att = hm.atten(X, A, st, wd, hm.STEEP[gate])
    out = {}
    for c in CLASSES:
        lo, hi = TARGET[gate][c]
        out[c] = {n: [round(float(np.mean((att[m] >= lo - 1e-6) & (att[m] <= hi + 1e-6))), 2),
                      [round(float(v), 2) for v in np.percentile(att[m], [10, 50, 90])]]
                  for n in hm.NS for m in [(C == c) & (Nn == n)]}
    return out


def main():
    P = hm.family_p(np.random.default_rng(26))
    X, A, C, Nn = labelled(P)
    win = {}
    for gate in ("resp", "learn"):
        win[gate], L = search(lambda s, w: gate_loss(X, A, C, Nn, s, w, gate))
        print(f"{gate:5s} alone : start {win[gate][0]} width {win[gate][1]}  loss {L:.4f}")
    win["shared"], L = search(lambda s, w: gate_loss(X, A, C, Nn, s, w, "resp") + gate_loss(X, A, C, Nn, s, w, "learn"))
    print(f"shared       : start {win['shared'][0]} width {win['shared'][1]}  loss {L:.4f}"
          f"  (0.4/0.4: {gate_loss(X, A, C, Nn, *hm.CUR, 'resp') + gate_loss(X, A, C, Nn, *hm.CUR, 'learn'):.4f})")
    res = {"windows": win, "targets": TARGET, "fit": {}, "R": [], "D": []}
    for tag in ("resp", "learn", "shared"):
        for gate in ("resp", "learn"):
            if tag != "shared" and tag != gate:
                continue
            res["fit"][f"{tag}/{gate}"] = in_target(X, A, C, Nn, *win[tag], gate)
    R, D = hm.family_r(), hm.family_d(np.random.default_rng(27))
    for r in R:
        if r["step"] != 500000 or r["beta"] not in (10, 20):
            continue
        e = {k: r[k] for k in ("ds", "cond", "N", "beta")}
        e["neff"] = round(float(np.median(r["neff"])), 3)
        for tag in ("resp", "learn", "shared"):
            for gate in ("resp", "learn"):
                if tag in (gate, "shared"):
                    e[f"{tag}/{gate}"] = round(float(np.median(hm.atten(r["h"], r["avg"], *win[tag], hm.STEEP[gate]))), 2)
        res["R"].append(e)
    for r in D:
        if r["view"] != "retina k0.8":
            continue
        e = {"ds": r["ds"], "cond": r["cond"], "h": [round(float(np.median(h)), 3) for h in r["hs"]]}
        hmin, amin = np.min(r["hs"], 0), np.min(r["avgs"], 0)
        for tag in ("resp", "shared"):
            e[f"{tag}/resp"] = round(float(np.median(hm.atten(hmin, amin, *win[tag], hm.STEEP["resp"]))), 2)
        for tag in ("learn", "shared"):
            e[f"{tag}/learn"] = [round(float(np.median(hm.atten(h, a, *win[tag], hm.STEEP["learn"]))), 2)
                                 for h, a in zip(r["hs"], r["avgs"])]
        res["D"].append(e)
    with open(f"{hm.W}/hoyer_fit2.json", "w") as f:
        json.dump(res, f, default=float)
    print(f"wrote {hm.W}/hoyer_fit2.json")


if __name__ == "__main__":
    main()
