#!/usr/bin/env python3
"""Map 1 - Hoyer (NSE as reference) on group-output-like vectors and dataset inputs,
and fit the dendrite noise-gate window (run from the worktree root).

Gate (hd_ngm2_dendrite.cpp): attenuation = 1 - g * (1 - s(avg, {0.5,0.5})),
  g = s((x - start) / width, {steep, 0.5}),  steep 0.25 (response) / 0.5 (learning).
The fit targets g (the shape part); the avg term is reported, not fitted.

Families
  P  parametric post-inhibition vectors: k = max(1, round(f N)) winners of amplitude
     a * (1 - U[0, 0.3]), the rest U[0, bg]. Classes (user, 2026-10-26: sparsity is a
     fraction of N; "many moderate winners" = neither signal nor noise):
       signal f <= 0.10 -> g <= 0.1,  middle 0.2 <= f <= 0.4 -> 0.3 <= g <= 0.7,
       noise  f >= 0.6  -> g >= 0.9.  Labelled only when bg <= a / 4.
  R  real 50-neuron pre-inhibition outputs (work/<ds>/resp), quantile-resampled to N
     (fraction-preserving), then max_softmax(beta).
  D  dataset inputs (first-layer dendrites): raw, and retina ON / OFF at k = 0.8.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "explore/nse_alt")
from ngm2_math import nse, sigmoid, max_softmax, retina_d, retina_out   # noqa: E402
from build_dnse import hoyer_noise, RAW                                 # noqa: E402

W = "explore/nse_alt/work"
NS = [50, 100, 200, 400]
FS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
AMPS = [0.1, 0.3, 0.6, 1.0]
BGS = [0.0, 0.02, 0.05, 0.1]
NSAMP = 64
BETAS = [5, 10, 20, 40]
R_DS = ["mnist", "kmnist", "emnist", "dsprites", "cifar10", "mnist_retina_k08", "chairs3d_retina_k08"]
R_STEPS = [50000, 500000]
CONDS = ["real", "perm", "unif"]
STEEP = {"resp": 0.25, "learn": 0.5}
TARGET = {"signal": (0.0, 0.1), "middle": (0.3, 0.7), "noise": (0.9, 1.0)}
CUR = (0.4, 0.4)


def neff_frac(V):
    l1, l2 = V.sum(-1, dtype=np.float64), (V.astype(np.float64) ** 2).sum(-1)
    return l1 * l1 / np.maximum(l2, 1e-300) / V.shape[-1]


def g(x, start, width, steep):
    return sigmoid((np.asarray(x, np.float32) - np.float32(start)) / np.float32(width), steep, 0.5)


def atten(x, avg, start, width, steep):
    return 1 - g(x, start, width, steep) * (1 - sigmoid(avg, 0.5, 0.5))


def cls(f, a, bg):
    if bg > a / 4:
        return None
    if f <= 0.10:
        return "signal"
    if 0.2 <= f <= 0.4:
        return "middle"
    if f >= 0.6:
        return "noise"
    return None


def family_p(rng):
    rows = []
    for N in NS:
        for f in FS:
            k = max(1, round(f * N))
            for a in AMPS:
                for bg in BGS:
                    V = rng.uniform(0, bg, (NSAMP, N)).astype(np.float32) if bg > 0 else np.zeros((NSAMP, N), np.float32)
                    for r in range(NSAMP):
                        idx = rng.choice(N, k, replace=False)
                        V[r, idx] = a * (1 - rng.uniform(0, 0.3, k))
                    rows.append(dict(N=N, f=f, k=k, a=a, bg=bg, cls=cls(f, a, bg),
                                     h=hoyer_noise(V).astype(np.float32), nse=nse(V), avg=V.mean(1)))
    return rows


def resample(O, N):
    """Fraction-preserving: each row's empirical quantile function sampled at N points."""
    S = np.sort(O, 1)
    u0 = (np.arange(O.shape[1]) + 0.5) / O.shape[1]
    u1 = (np.arange(N) + 0.5) / N
    return np.stack([np.interp(u1, u0, s) for s in S]).astype(np.float32)


def family_r():
    out = []
    for ds in R_DS:
        for s in R_STEPS:
            for c in CONDS:
                O = np.load(f"{W}/{ds}/resp/{c}_{s}.npz")["out"].astype(np.float32)[:500]
                for N in NS:
                    V0 = resample(O, N)
                    if N == 50:   # identity check: resampling at N = 50 must not change the measure
                        assert np.allclose(hoyer_noise(V0), hoyer_noise(O), atol=1e-5)
                    for b in [0] + BETAS:
                        V = V0 if b == 0 else max_softmax(V0, b)
                        out.append(dict(ds=ds, step=s, cond=c, N=N, beta=b, h=hoyer_noise(V).astype(np.float32),
                                        nse=nse(V), avg=V.mean(1), neff=neff_frac(V)))
    return out


def family_d(rng):
    out = []
    for ds in ["mnist", "kmnist", "emnist", "dsprites", "cifar10", "chairs3d"]:
        for c in CONDS:
            X = np.load(f"{W}/{ds}/resp/{c}_500000.npz")["inp"].astype(np.float32)[:500]
            out.append(dict(ds=ds, cond=c, view="raw", hs=[hoyer_noise(X)], ns=[nse(X)], avgs=[X.mean(1)]))
            if ds != "emnist":
                d = retina_d(X, RAW[ds], 0.8)
                on, off = retina_out(d, "on", rng), retina_out(d, "off", rng)
                out.append(dict(ds=ds, cond=c, view="retina k0.8", hs=[hoyer_noise(on), hoyer_noise(off)],
                                ns=[nse(on), nse(off)], avgs=[on.mean(1), off.mean(1)]))
    return out


def fit(P):
    lab = [r for r in P if r["cls"]]
    X = np.concatenate([r["h"] for r in lab])
    C = np.concatenate([[r["cls"]] * len(r["h"]) for r in lab])
    Nn = np.concatenate([[r["N"]] * len(r["h"]) for r in lab])
    groups = [(C == c) & (Nn == n) for c in TARGET for n in NS]

    def loss(st, wd):
        tot = 0.0
        for steep in STEEP.values():
            G = g(X, st, wd, steep)
            lo = np.array([TARGET[c][0] for c in C], np.float32)
            hi = np.array([TARGET[c][1] for c in C], np.float32)
            v = np.maximum(lo - G, 0) + np.maximum(G - hi, 0)
            tot += np.mean([np.mean(v[m] ** 2) for m in groups])   # equal weight per (class, N)
        return tot

    best = (np.inf, None)
    for st in np.arange(-0.5, 1.0, 0.02):
        for wd in np.arange(0.05, 1.5, 0.02):
            L = loss(st, wd)
            if L < best[0]:
                best = (L, (st, wd))
    s0, w0 = best[1]
    for st in np.arange(s0 - 0.03, s0 + 0.03, 0.005):
        for wd in np.arange(max(0.02, w0 - 0.03), w0 + 0.03, 0.005):
            L = loss(st, wd)
            if L < best[0]:
                best = (L, (round(float(st), 3), round(float(wd), 3)))
    return best[1], best[0], loss(*CUR)


def pct(x, q=(10, 50, 90)):
    return [round(float(v), 3) for v in np.percentile(x, q)]


def main():
    rng = np.random.default_rng(26)
    P, R, D = family_p(rng), family_r(), family_d(rng)
    (st, wd), L, Lcur = fit(P)
    print(f"fitted window start {st}  width {wd}  loss {L:.4f}   (current 0.4/0.4 loss {Lcur:.4f})")
    for name, steep in STEEP.items():
        z = 2.197 / (1 / (1 - steep ** 0.1))
        print(f"  {name}: g 10-90% band x in [{st + wd * (0.5 - z):.3f}, {st + wd * (0.5 + z):.3f}]")
    res = {"window": [st, wd], "loss": L, "loss_current": Lcur, "P": [], "R": [], "D": []}
    for r in P:
        e = {k: r[k] for k in ("N", "f", "k", "a", "bg", "cls")}
        e["h"], e["nse"] = pct(r["h"]), pct(r["nse"])
        for tag, (s_, w_) in (("fit", (st, wd)), ("cur", CUR)):
            for gn, steep in STEEP.items():
                e[f"g_{gn}_{tag}"] = round(float(np.median(g(r["h"], s_, w_, steep))), 3)
                e[f"att_{gn}_{tag}"] = round(float(np.median(atten(r["h"], r["avg"], s_, w_, steep))), 3)
        res["P"].append(e)
    for r in R:
        e = {k: r[k] for k in ("ds", "step", "cond", "N", "beta")}
        e["h"], e["nse"], e["neff"], e["avg"] = pct(r["h"]), pct(r["nse"]), pct(r["neff"]), pct(r["avg"])
        for tag, (s_, w_) in (("fit", (st, wd)), ("cur", CUR)):
            for gn, steep in STEEP.items():
                e[f"att_{gn}_{tag}"] = pct(atten(r["h"], r["avg"], s_, w_, steep))
        res["R"].append(e)
    for r in D:
        e = {k: r[k] for k in ("ds", "cond", "view")}
        e["h"] = [pct(h) for h in r["hs"]]
        e["nse"] = [pct(n) for n in r["ns"]]
        hmin, amin = np.min(r["hs"], 0), np.min(r["avgs"], 0)
        for tag, (s_, w_) in (("fit", (st, wd)), ("cur", CUR)):
            e[f"att_resp_{tag}"] = pct(atten(hmin, amin, s_, w_, STEEP["resp"]))
            e[f"att_learn_{tag}"] = [pct(atten(h, a, s_, w_, STEEP["learn"])) for h, a in zip(r["hs"], r["avgs"])]
        res["D"].append(e)
    with open(f"{W}/hoyer_map.json", "w") as f:
        json.dump(res, f, default=float)
    print(f"wrote {W}/hoyer_map.json")


if __name__ == "__main__":
    main()
