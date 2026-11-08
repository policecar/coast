#!/usr/bin/env python3
"""Reduce work/<ds>/resp/*.npz to the artefact's data file (work/explorer_data.json).

Vector kinds:
  pre    group activity before inhibition (the pass-through probe output)
  post   soft_wta(pre, S) via the validated mirror in ngm2_math.py
  input  dataset input vectors (final checkpoint only; inputs do not depend on
         the training stage), plus the image-grid measure nbr_corr

Per (dataset, kind, step, cond): quartiles of every measure over all N vectors,
and float32 per-vector values for a fixed random sample (histograms, scatter).
Per (dataset, kind, step): AUC of real vs perm, real vs unif (same step) and
real@step vs real@first step, all on the full N.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

import measures as M
from ngm2_math import soft_wta

WORK = os.path.join(os.path.dirname(__file__), "work")
INPUT_STEP = 500000
CONDS = ["real", "perm", "unif"]
DATASETS = ["mnist", "emnist", "kmnist", "cifar10", "dsprites", "chairs3d"]
SHAPES = {"mnist": (28, 28, 1), "emnist": (28, 28, 1), "kmnist": (28, 28, 1),
          "cifar10": (32, 32, 3), "dsprites": (64, 64, 1), "chairs3d": (64, 64, 3)}
STRENGTH = 5.0
N_SAMPLE = 160
N_EXAMPLE = 6
MEASURE_KEYS = list(M.MEASURES) + ["mean", "max"]


def all_measures(V, shape=None):
    V = V.astype(np.float64)
    m = M.compute_all(V)
    m["mean"] = V.mean(1)
    m["max"] = V.max(1)
    if shape is not None:
        m["nbr_corr"] = M.nbr_corr(V, shape)
    return m


def b64f32(a):
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


def b64u8(a):
    return base64.b64encode(np.clip(np.rint(np.asarray(a) * 255), 0, 255)
                            .astype(np.uint8).tobytes()).decode()


def png_uri(vec, shape):
    h, w, c = shape
    img = np.clip(np.rint(vec.reshape(h, w, c) * 255), 0, 255).astype(np.uint8)
    im = Image.fromarray(img[..., 0] if c == 1 else img)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def summarise(m, sample_idx):
    return {
        "q": {k: [float(x) for x in np.percentile(v, [10, 25, 50, 75, 90])] for k, v in m.items()},
        "s": {k: b64f32(v[sample_idx]) for k, v in m.items()},
    }


def aucs(a, b):
    return {k: float(M.auc(a[k], b[k])) for k in a}


def available_steps(ds):
    """Checkpoints with all three probe conditions present (the schedule differs per dataset)."""
    d = f"{WORK}/{ds}/resp"
    if not os.path.isdir(d):
        return []
    steps = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(d) if f.endswith(".npz")}
    return sorted(s for s in steps if all(os.path.exists(f"{d}/{c}_{s}.npz") for c in CONDS))


def build_dataset(ds, rng, steps):
    load = lambda c, s: np.load(f"{WORK}/{ds}/resp/{c}_{s}.npz")
    out = {"kinds": {"pre": {}, "post": {}}, "auc": {"pre": {}, "post": {}}}
    first = {}
    for s in steps:
        meas = {"pre": {}, "post": {}}
        for c in CONDS:
            pre = load(c, s)["out"]
            sample = rng.choice(len(pre), N_SAMPLE, replace=False)
            ex = sample[:N_EXAMPLE]
            for kind, V in (("pre", pre), ("post", soft_wta(pre, STRENGTH))):
                m = all_measures(V)
                meas[kind][c] = m
                g = summarise(m, sample)
                g["ex"] = b64u8(V[ex])
                out["kinds"][kind][f"{s}|{c}"] = g
        for kind in ("pre", "post"):
            if s == steps[0]:
                first[kind] = meas[kind]["real"]
            r = meas[kind]["real"]
            out["auc"][kind][str(s)] = {
                "perm": aucs(r, meas[kind]["perm"]),
                "unif": aucs(r, meas[kind]["unif"]),
                "early": aucs(r, first[kind]),
            }
    # inputs at INPUT_STEP (they do not depend on the training stage)
    shape = SHAPES[ds]
    imeas, out["kinds"]["input"], out["images"] = {}, {}, {}
    for c in CONDS:
        X = load(c, INPUT_STEP)["inp"]
        sample = rng.choice(len(X), N_SAMPLE, replace=False)
        m = all_measures(X, shape)
        imeas[c] = m
        out["kinds"]["input"][f"{INPUT_STEP}|{c}"] = summarise(m, sample)
        out["images"][c] = [png_uri(X[i], shape) for i in sample[:4]]
    out["auc"]["input"] = {str(INPUT_STEP): {
        "perm": aucs(imeas["real"], imeas["perm"]),
        "unif": aucs(imeas["real"], imeas["unif"]),
    }}
    out["steps"] = steps
    out["input_n"] = int(np.prod(shape))
    out["shape"] = list(shape)
    return out


def main():
    rng = np.random.default_rng(7)
    data = {"input_step": INPUT_STEP, "conds": CONDS, "strength": STRENGTH, "n_sample": N_SAMPLE,
            "n_example": N_EXAMPLE, "measures": MEASURE_KEYS + ["nbr_corr"], "datasets": {}}
    for ds in DATASETS:
        steps = available_steps(ds)
        if not steps or steps[0] != 1 or INPUT_STEP not in steps:
            print(f"skip {ds}: incomplete ({steps})")
            continue
        data["datasets"][ds] = build_dataset(ds, rng, steps)
        print(f"built {ds}: {len(steps)} checkpoints, last {steps[-1]}")
    path = f"{WORK}/explorer_data.json"
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(path, os.path.getsize(path) / 1e6, "MB")


if __name__ == "__main__":
    main()
