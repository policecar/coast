#!/usr/bin/env python3
"""Compare measures.js against measures.py on real response vectors and inputs.

The NSE variants are excluded from the pass gate: measures.py mirrors the
model's float32 NSE, measures.js computes it in double (expected ~2e-7)."""
import json
import subprocess
import sys

import numpy as np

import measures as M

rng = np.random.default_rng(0)
vecs = [np.load(f"work/mnist/resp/{c}_{s}.npz")["out"][:40] for c in ("real", "perm", "unif")
        for s in (1, 500000)]
vecs.append(np.load("work/mnist/resp/real_500000.npz")["inp"][:20])
vecs.append(rng.random((20, 300)))
ref = {}
for V in vecs:
    for k, v in M.compute_all(V).items():
        ref.setdefault(k, []).extend(v.tolist())
ref["mean"] = [float(x) for V in vecs for x in V.astype(np.float64).mean(1)]
ref["max"] = [float(x) for V in vecs for x in V.max(1)]
payload = json.dumps([v.astype(float).tolist() for V in vecs for v in V])
js = subprocess.run(["node", "-e", """
const {MEASURE_FNS}=require('./measures.js'); let d='';
process.stdin.on('data',c=>d+=c).on('end',()=>{const V=JSON.parse(d);const o={};
for(const k in MEASURE_FNS) o[k]=V.map(v=>MEASURE_FNS[k](v)); console.log(JSON.stringify(o));});"""],
                    input=payload, capture_output=True, text=True, check=True)
out = json.loads(js.stdout)
worst = 0.0
for k in ref:
    a, b = np.array(ref[k]), np.array(out[k])
    rel = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-9))
    worst = max(worst, rel if k not in ("nse", "nse_minsub", "nse_medsub") else 0.0)
    print(f"{k:13s} max rel diff {rel:.2e}")
sys.exit(0 if worst < 1e-9 else 1)
