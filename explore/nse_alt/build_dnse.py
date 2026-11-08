#!/usr/bin/env python3
"""Data for the dendrite-noise-gate artefact (work/dnse_data.json).

Inputs: work/<ds>/resp/<cond>_500000.npz ("inp" = probe inputs incl. the dataset's
U(0,0.05) noise; "out" = pre-inhibition neuron outputs of the trained group).
Retina outputs are computed with ngm2_math.retina_d/retina_out (validated against
retina::contrast_filter_t to <= 7e-7).

The two dendrite gates (hd_ngm2_dendrite.cpp):
  response (get_response, :304):   1 - s((nse-0.8)/0.2, {0.25,0.5}) * (1 - s(avg, {0.5,0.5}))
                                   nse, avg = min over the dendrite's input channels
  learning (adapt_synapses, :410): 1 - s((nse-0.8)/0.2, {0.5,0.5})  * (1 - s(avg, {0.5,0.5}))
                                   per input channel
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

from ngm2_math import nse, sigmoid, retina_d, retina_out

WORK = os.path.join(os.path.dirname(__file__), "work")
STEP = 500000
CONDS = ["real", "perm", "unif"]
RAW = {"mnist": (28, 28, 1), "emnist": (28, 28, 1), "kmnist": (28, 28, 1), "cifar10": (32, 32, 3),
       "dsprites": (64, 64, 1), "chairs3d": (64, 64, 3), "chairs3d_inv": (64, 64, 3)}
RET = ["mnist", "kmnist", "cifar10", "dsprites", "chairs3d"]
KS = [0.5, 0.7, 0.8, 0.9, 1.0]
N_SCATTER = 250
N_RET = 500
NEURON_NOISE_MEAN = 0.03     # default_neuron_noise_range {0.01, 0.05}


def gate(n, a, steep):
    return 1 - sigmoid((n - 0.8) / 0.2, steep, 0.5) * (1 - sigmoid(a, 0.5, 0.5))


def sep(a, b):
    """2*AUC - 1 = P(a > b) - P(a < b)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    allv = np.concatenate([a, b])
    r = allv.argsort(kind="stable").argsort() + 1.0
    auc = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return float(2 * auc - 1)


def hoyer_noise(V):
    n = V.shape[1]
    l1, l2 = V.sum(1), np.sqrt((V.astype(np.float64) ** 2).sum(1))
    return 1 - (np.sqrt(n) - l1 / np.maximum(l2, 1e-12)) / (np.sqrt(n) - 1)


# every measure oriented so that HIGH = "looks like noise"
MEASURES = {
    "NSE": lambda V: nse(V),
    "NSE above floor": lambda V: nse(np.clip(V - np.median(V, 1, keepdims=True), 0, None)),
    "1 - top-10% share": lambda V: 1 - np.sort(V, 1)[:, -max(1, V.shape[1] // 10):].sum(1) / np.maximum(V.sum(1), 1e-12),
    "mean / max": lambda V: V.mean(1) / np.maximum(V.max(1), 1e-12),
    "1 - Hoyer": hoyer_noise,
}


def r3(x):
    return float(round(float(x), 3))


def png(vec, shape, signed=False):
    h, w, c = shape
    a = np.asarray(vec, np.float32).reshape(h, w, c)
    img = np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img[..., 0] if c == 1 else img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load(ds, cond, key):
    return np.load(f"{WORK}/{ds}/resp/{cond}_{STEP}.npz")[key].astype(np.float32)


def raw_section(rng):
    out = {}
    for ds in RAW:
        d = {"scatter": {}, "stats": {}}
        pk, pku, gr = {}, {}, {}
        for c in CONDS:
            X, O = load(ds, c, "inp"), load(ds, c, "out")
            n, a = nse(X), X.mean(1)
            g_resp, g_learn = gate(n, a, 0.25), gate(n, a, 0.5)
            U = np.clip((O - NEURON_NOISE_MEAN) / g_resp[:, None], 0, None)
            pk[c], pku[c], gr[c] = O.max(1), U.max(1), g_resp
            idx = rng.choice(len(X), N_SCATTER, replace=False)
            d["scatter"][c] = [[r3(n[i]), r3(a[i])] for i in idx]
            d["stats"][c] = {"nse": r3(np.median(n)), "avg": r3(np.median(a)), "gate_resp": r3(np.median(g_resp)),
                             "gate_learn": r3(np.median(g_learn)), "peak": r3(np.median(pk[c])),
                             "peak_ungated": r3(np.median(pku[c]))}
        d["sep"] = {f"{x}_vs_{y}": {"with_gate": r3(sep(pk[x], pk[y])), "without_gate": r3(sep(pku[x], pku[y])),
                                    "gate_itself": r3(sep(gr[x], gr[y]))}
                    for x, y in (("real", "perm"), ("real", "unif"))}
        d["saturated"] = bool(np.median(pk["real"]) > 0.99)
        out[ds] = d
    return out


def sparsity_section(rng):
    fs = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nse_curves = {}
    for N in (784, 3072, 4096, 12288):
        for eps in (0.0, 0.05):
            vals = []
            for f in fs:
                V = np.zeros((40, N), np.float32)
                V[:, :max(1, int(round(f * N)))] = 1.0
                V = np.clip(V + rng.uniform(0, eps, V.shape).astype(np.float32), 0, 1)
                vals.append(r3(np.median(nse(V))))
            nse_curves[f"{N}|{eps}"] = vals
    meas = {}
    for m, fn in MEASURES.items():
        vals = []
        for f in fs:
            V = np.zeros((40, 4096), np.float32)
            V[:, :max(1, int(round(f * 4096)))] = 1.0
            V = np.clip(V + rng.uniform(0, 0.05, V.shape).astype(np.float32), 0, 1)
            vals.append(r3(np.median(fn(V))))
        meas[m] = vals
    return {"fs": fs, "nse": nse_curves, "measures": meas}


def retina_section(rng):
    out = {}
    for ds in RET:
        shape = RAW[ds]
        oshape = (shape[0], shape[1], 1) if shape[2] == 3 else shape
        X = {c: load(ds, c, "inp")[:N_RET] for c in CONDS}
        ex_idx = {c: int(rng.integers(len(X[c]))) for c in CONDS}
        d = {"k": {}, "examples": {c: {"input": png(X[c][ex_idx[c]], shape)} for c in CONDS}}
        for k in KS:
            kk = {"scatter": {}, "stats": {}, "sep": {}, "img": {}}
            J = {}
            for c in CONDS:
                dd = retina_d(X[c], shape, k)
                on, off = retina_out(dd, "on", rng), retina_out(dd, "off", rng)
                n_on, n_off, a_on, a_off = nse(on), nse(off), on.mean(1), off.mean(1)
                g_joint = gate(np.minimum(n_on, n_off), np.minimum(a_on, a_off), 0.25)
                J[c] = g_joint
                idx = rng.choice(len(on), 150, replace=False)
                kk["scatter"][c] = [[r3(n_on[i]), r3(n_off[i])] for i in idx]
                kk["stats"][c] = {"nse_on": r3(np.median(n_on)), "nse_off": r3(np.median(n_off)),
                                  "learn_on": r3(np.median(gate(n_on, a_on, 0.5))),
                                  "learn_off": r3(np.median(gate(n_off, a_off, 0.5))),
                                  "resp_joint": r3(np.median(g_joint))}
                i = ex_idx[c]
                kk["img"][c] = {"on": png(on[i], oshape), "off": png(off[i], oshape),
                                "nse_on": r3(n_on[i]), "nse_off": r3(n_off[i])}
            kk["sep"] = {"real_vs_perm": r3(sep(J["real"], J["perm"])), "real_vs_unif": r3(sep(J["real"], J["unif"]))}
            d["k"][str(k)] = kk
        out[ds] = d
    return out


def candidates_section(rng):
    """Separation (noise > real) of each measure: raw input, retina ON / OFF alone,
    and the ON/OFF pair judged jointly (min over channels), at k = 0.9."""
    rows = []
    for ds in RET:
        shape = RAW[ds]
        X = {c: load(ds, c, "inp")[:N_RET] for c in CONDS}
        R = {c: retina_d(X[c], shape, 0.9) for c in CONDS}
        ON = {c: retina_out(R[c], "on", rng) for c in CONDS}
        OFF = {c: retina_out(R[c], "off", rng) for c in CONDS}
        for view in ("raw", "ON", "OFF", "joint"):
            row = {"ds": ds, "view": view, "vals": {}}
            for m, fn in MEASURES.items():
                if view == "raw":
                    v = {c: fn(X[c]) for c in CONDS}
                elif view == "ON":
                    v = {c: fn(ON[c]) for c in CONDS}
                elif view == "OFF":
                    v = {c: fn(OFF[c]) for c in CONDS}
                else:
                    v = {c: np.minimum(fn(ON[c]), fn(OFF[c])) for c in CONDS}
                row["vals"][m] = [r3(sep(v["perm"], v["real"])), r3(sep(v["unif"], v["real"]))]
            rows.append(row)
    return {"measures": list(MEASURES), "rows": rows}


def main():
    rng = np.random.default_rng(23)
    data = {"step": STEP, "conds": CONDS, "raw": raw_section(rng), "sparsity": sparsity_section(rng),
            "retina": retina_section(rng), "candidates": candidates_section(rng), "ks": KS}
    path = f"{WORK}/dnse_data.json"
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(path, round(os.path.getsize(path) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
