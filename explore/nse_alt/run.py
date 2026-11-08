#!/usr/bin/env python3
"""Train ngm2 per dataset with log-spaced checkpoints, then probe every
checkpoint frozen with a pass-through inhibitor on real / perm / unif inputs.

Run from the worktree root (dataset paths in configs are CWD-relative):
    python3 explore/nse_alt/run.py [--datasets mnist,emnist] [--jobs 6]
    python3 explore/nse_alt/run.py --datasets mnist --tag _msm10 \
        --train-inh '{"type": "max_softmax", "strength": 10}'
    python3 explore/nse_alt/run.py --datasets mnist:mnist_retina --tag _hoyer \
        --retina-set '{"surround_weight": 0.8}' \
        --dendrite-set '{"noise_measure": "hoyer", "noise_window_start": 0.4, "noise_window_width": 0.4}'
A dataset entry "ds:cfg" trains configs/<cfg>.json (probe data still from <ds>)
into work/<cfg><tag>.

Layout under explore/nse_alt/work/<ds><tag>/:
    cfg/{train,real,perm,unif}.json
    ck/ck_<step>.h5                    chained training checkpoints
    resp/<cond>_<step>.npz             out: (N,50) pre-inhibition activity
                                       inp: (N,in) input vectors (final ck only)
    run.log
Resumable: existing checkpoints / npz files are skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_cfg import derive  # noqa: E402

COAST = "./build-min/coast"
WORK = "explore/nse_alt/work"
NOISE = f"{WORK}/noise"
STEPS = [1, 500, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000]
# input vectors are stored at this step only (they do not depend on training)
INPUT_STEP = 500000
PROBE_STEPS = 2000
DATASETS = ["mnist", "emnist", "kmnist", "cifar10", "dsprites", "chairs3d"]
CONDS = ["real", "perm", "unif"]


def overrides(ds, cond):
    if cond == "real":
        return {"use_test_set": True} if ds == "cifar10" else {}
    d = f"./{NOISE}/{ds}_{cond}"
    if ds in ("mnist", "emnist", "kmnist"):
        return {"image_file": f"{d}/images-idx3-ubyte", "label_file": f"{d}/labels-idx1-ubyte"}
    if ds == "cifar10":
        return {"data_dir": d, "use_test_set": True}
    if ds == "dsprites":
        return {"hdf5_file": f"{d}/imgs.hdf5"}
    if ds == "chairs3d":
        return {"data_dir": d}
    raise ValueError(ds)


def sh(cmd, log):
    t0 = time.time()
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"# exit {r.returncode} in {time.time() - t0:.1f}s\n")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd} failed ({r.returncode}); see {log}")


def extract(h5path, npz, keep_input):
    with h5py.File(h5path, "r") as f:
        b = f["steps/buffers"]
        sizes = f["metadata/topology/buffer_sizes"][:]
        ids = [int(x) for x in f["metadata/topology/buffer_ids"][:]]
        # node 0 = dataset; the (single) group is the only 50-wide buffer;
        # retina configs add ON/OFF nodes 10/11. A retina adds one step of
        # latency, so inp[t] is not the image the group answered at row t.
        groups = [i for i, n in zip(ids, sizes) if n == 50]
        assert len(groups) == 1 and 0 in ids, (ids, sizes)
        inp = b["0"]["data"][:]
        out = b[str(groups[0])]["data"][:]
        ret = {k: b[str(i)]["data"][:] for k, i in (("ret_on", 10), ("ret_off", 11)) if i in ids}
    assert inp.shape[1] == sizes[ids.index(0)], (inp.shape, sizes)
    # neuron_t::get_response clamps activity to [0,1], so the pass-through's
    # clamp(raw,0,1) is an exact identity; a value outside means it is not.
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1, \
        "pass-through identity needs raw in [0,1]"
    arrs = {"out": out.astype(np.float32)}
    if keep_input:
        arrs["inp"] = inp.astype(np.float32)
        arrs.update({k: v.astype(np.float32) for k, v in ret.items()})
    np.savez_compressed(npz, **arrs)


def run_dataset(ds, steps, probe_steps, work, tag="", train_inh=None, ds_set=None,
                dendrite_set=None, retina_set=None):
    ds, _, cfg_name = ds.partition(":")
    cfg_name = cfg_name or ds
    w = f"{work}/{cfg_name}{tag}"
    for sub in ("cfg", "ck", "resp"):
        os.makedirs(f"{w}/{sub}", exist_ok=True)
    log = f"{w}/run.log"
    with open(f"configs/{cfg_name}.json") as f:
        base = json.load(f)
    for entry in base["datasets"][ds].values():   # training and probes alike
        entry.update(ds_set or {})
    for d in base["ngm2"]["dendrites"].values():
        d.update(dendrite_set or {})
    for r in base.get("retina", {}).values():
        r.update(retina_set or {})
    cfgs = {"train": derive(base, "train", train_inh=train_inh)}
    for c in CONDS:
        cfgs[c] = derive(base, "probe", 1000, overrides(ds, c), train_inh)
    if train_inh is not None:
        cfgs["real_inh"] = derive(base, "inh", 1000, overrides(ds, "real"), train_inh)
    for k, v in cfgs.items():
        with open(f"{w}/cfg/{k}.json", "w") as f:
            json.dump(v, f, indent=1)

    prev = None
    for s in steps:
        ck = f"{w}/ck/ck_{s}.h5"
        if not os.path.exists(ck):
            cmd = [COAST, "-n", str(s - (prev or 0))]
            if prev is not None:
                cmd += ["--load-state", f"{w}/ck/ck_{prev}.h5"]
            sh(cmd + ["--save-state", ck, f"{w}/cfg/train.json"], log)
        for c in CONDS:
            npz = f"{w}/resp/{c}_{s}.npz"
            if os.path.exists(npz):
                continue
            tmp = f"{w}/resp/{c}_{s}.h5"
            if os.path.exists(tmp):
                os.remove(tmp)
            sh([COAST, "-n", str(probe_steps), "--load-state", ck,
                "--log-buffers", tmp, f"{w}/cfg/{c}.json"], log)
            extract(tmp, npz, keep_input=(s == INPUT_STEP))
            os.remove(tmp)
        prev = s
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="tiny run into work/smoke_*")
    ap.add_argument("--extra-steps", default="", help="comma list of checkpoints beyond the default schedule")
    ap.add_argument("--tag", default="", help="suffix of the work directory, e.g. _msm10")
    ap.add_argument("--train-inh", default=None, help="JSON inhibition params used for training")
    ap.add_argument("--ds-set", default=None, help="JSON merged into the dataset entries")
    ap.add_argument("--dendrite-set", default=None, help="JSON merged into every ngm2 dendrite config")
    ap.add_argument("--retina-set", default=None, help="JSON merged into every retina config")
    a = ap.parse_args()
    # passed explicitly: worker processes re-import this module (forkserver is
    # the Linux default since Python 3.14), so mutated globals never reach them
    steps, probe_steps, work = ([1, 50], 50, f"{WORK}/smoke") if a.smoke \
        else (STEPS, PROBE_STEPS, WORK)
    if a.extra_steps:
        steps = sorted(set(steps) | {int(x) for x in a.extra_steps.split(",")})
    dss = a.datasets.split(",")
    with ProcessPoolExecutor(a.jobs) as ex:
        train_inh = json.loads(a.train_inh) if a.train_inh else None
        ds_set = json.loads(a.ds_set) if a.ds_set else None
        den_set = json.loads(a.dendrite_set) if a.dendrite_set else None
        ret_set = json.loads(a.retina_set) if a.retina_set else None
        futs = {ex.submit(run_dataset, d, steps, probe_steps, work, a.tag, train_inh, ds_set, den_set, ret_set): d
                for d in dss}
        for fu, d in futs.items():
            try:
                fu.result()
                print(f"done {d}", flush=True)
            except Exception as e:  # report every dataset, fail at the end
                print(f"FAILED {d}: {e}", flush=True)


if __name__ == "__main__":
    main()
