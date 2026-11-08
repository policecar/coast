#!/usr/bin/env python3
"""Reduce the max_softmax training runs to the data file of the second artefact
(work/msm_data.json).

Runs (work/<ds><tag>/resp/<cond>_<step>.npz, pre-inhibition, pass-through probe):
  swta   tag ""        trained with the default soft_wta (strength 5)
  msm10  tag "_msm10"  trained with max_softmax, beta 10
  msm20  tag "_msm20"  trained with max_softmax, beta 20
The group output a run emits is its own inhibitor applied to the pre vector,
computed with the validated mirrors in ngm2_math.py.

Per (dataset, run, cond, step): medians of max / mean / n_eff (= sum/max) of the
pre and post vectors, and the fraction of neurons the inhibitor raised.
At FINAL_STEP: a set of example inputs (thumbnail + per-run pre vectors) for the
page's explorer and, for mnist, the winner-by-digit table of every run.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

from ngm2_math import max_softmax, soft_wta

WORK = os.path.join(os.path.dirname(__file__), "work")
FINAL_STEP = 500000
CONDS = ["real", "perm", "unif"]
RUNS = {"swta": ("", ("soft_wta", 5.0)), "msm10": ("_msm10", ("max_softmax", 10.0)),
        "msm20": ("_msm20", ("max_softmax", 20.0))}
SHAPES = {"mnist": (28, 28, 1), "chairs3d": (64, 64, 3), "chairs3d_inv": (64, 64, 3)}  # _inv: loader invert=true
N_PER_CLASS = 12   # mnist examples per digit
N_REAL = 36        # chairs3d real examples
N_NOISE = 12       # examples per noise condition


def inhibit(kind, param, pre):
    return soft_wta(pre, param) if kind == "soft_wta" else max_softmax(pre, param)


def load(ds, tag, cond, step, key="out"):
    return np.load(f"{WORK}/{ds}{tag}/resp/{cond}_{step}.npz")[key]


def steps_of(ds, tag):
    d = f"{WORK}/{ds}{tag}/resp"
    s = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(d) if f.endswith(".npz")}
    return sorted(x for x in s if all(os.path.exists(f"{d}/{c}_{x}.npz") for c in CONDS))


def stats(V):
    mx = V.max(1)
    return {"max": float(np.median(mx)), "mean": float(np.median(V.mean(1))),
            "neff": float(np.median(V.sum(1) / np.maximum(mx, 1e-12)))}


def b64u16(a):
    return base64.b64encode(np.clip(np.rint(np.asarray(a) * 65535), 0, 65535)
                            .astype("<u2").tobytes()).decode()


def png_uri(vec, shape):
    h, w, c = shape
    img = np.clip(np.rint(vec.reshape(h, w, c) * 255), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img[..., 0] if c == 1 else img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mnist_labels(inp):
    """Labels of probed inputs by nearest training image (the probe adds U(0,0.05)
    noise; matches are checked to be unambiguous)."""
    X = np.frombuffer(open("data/mnist/train-images-idx3-ubyte", "rb").read(), np.uint8,
                      offset=16).reshape(-1, 784).astype(np.float32) / 255
    lab = np.frombuffer(open("data/mnist/train-labels-idx1-ubyte", "rb").read(), np.uint8, offset=8)
    d = (inp ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * inp @ X.T
    o = np.argsort(d, 1)[:, :2]
    r = np.arange(len(d))
    assert (d[r, o[:, 1]] / d[r, o[:, 0]]).min() > 1.5, "ambiguous label match"
    return lab[o[:, 0]].astype(int)


def nmi(winner, label, n_win, n_lab):
    """I(winner; label) / H(label) over the joint counts."""
    J = np.zeros((n_lab, n_win))
    np.add.at(J, (label, winner), 1)
    P = J / J.sum()
    pl, pw = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / (pl @ pw)[nz])).sum()
    hl = -(pl[pl > 0] * np.log(pl[pl > 0])).sum()
    return J.astype(int), float(mi / hl)


def build(ds, rng):
    steps = sorted(set.intersection(*(set(steps_of(ds, t)) for t, _ in RUNS.values())))
    assert FINAL_STEP in steps, (ds, steps)
    out = {"steps": steps, "shape": list(SHAPES[ds]), "traj": {}, "runs": list(RUNS)}
    for run, (tag, (kind, p)) in RUNS.items():
        tr = out["traj"][run] = {}
        for c in CONDS:
            rows = {"pre": [], "post": [], "raised": []}
            for s in steps:
                pre = load(ds, tag, c, s).astype(np.float32)
                post = inhibit(kind, p, pre)
                rows["pre"].append(stats(pre))
                rows["post"].append(stats(post))
                rows["raised"].append(float(np.mean(post > pre + 1e-6)))
            tr[c] = {k: ({m: [r[m] for r in v] for m in v[0]} if k != "raised" else v)
                     for k, v in rows.items()}

    # explorer at FINAL_STEP. The probe continues the training input stream, and
    # all runs share the dataset seed, so they see the same images; checked here.
    inp = {c: load(ds, "", c, FINAL_STEP, "inp") for c in CONDS}
    for run, (tag, _) in RUNS.items():
        for c in CONDS:
            other = load(ds, tag, c, FINAL_STEP, "inp")
            # same images, independent U(0,0.05) noise draws
            assert np.abs(other - inp[c]).max() <= 0.0501, (run, c, np.abs(other - inp[c]).max())
    ex = []
    if ds == "mnist":
        labels = mnist_labels(inp["real"])
        for k in range(10):
            ex += [("real", int(i), k) for i in rng.choice(np.flatnonzero(labels == k), N_PER_CLASS, replace=False)]
    else:
        labels = None
        ex += [("real", int(i), -1) for i in rng.choice(len(inp["real"]), N_REAL, replace=False)]
    for c in ("perm", "unif"):
        ex += [(c, int(i), -1) for i in rng.choice(len(inp[c]), N_NOISE, replace=False)]
    out["examples"] = [{"cond": c, "label": k, "img": png_uri(inp[c][i], SHAPES[ds])} for c, i, k in ex]
    out["pre"] = {}
    for run, (tag, _) in RUNS.items():
        V = {c: load(ds, tag, c, FINAL_STEP).astype(np.float32) for c in CONDS}
        out["pre"][run] = b64u16(np.stack([V[c][i] for c, i, _ in ex]))
    if labels is not None:
        out["winners"] = {}
        for run, (tag, (kind, p)) in RUNS.items():
            post = inhibit(kind, p, load(ds, tag, "real", FINAL_STEP).astype(np.float32))
            J, score = nmi(post.argmax(1), labels, 50, 10)
            out["winners"][run] = {"counts": J.tolist(), "nmi": score,
                                   "used": int((J.sum(0) > 0).sum())}
    return out


def main():
    rng = np.random.default_rng(11)
    data = {"final_step": FINAL_STEP, "conds": CONDS,
            "runs": {k: {"tag": t, "kind": kind, "param": p} for k, (t, (kind, p)) in RUNS.items()},
            "datasets": {ds: build(ds, rng) for ds in SHAPES}}
    path = f"{WORK}/msm_data.json"
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(path, round(os.path.getsize(path) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
