#!/usr/bin/env python3
"""Derive train/probe configs from a COAST ngm2 config.

train:  the config unchanged.
probe:  every ngm2 group frozen, inhibitor = pass-through (random_index with
        ri_strength 0 => T = 0 => out = clamp(raw, 0, 1); raw activities stay
        inside [0,1], verified), input seeds offset so the probe draws a
        different input stream than training, optional per-dataset-entry
        overrides (noise datasets, cifar test split).
inh:    as probe but with the training inhibitor `train_inh` (validates the
        Python mirror of that inhibitor against the model).

train_inh (all modes): inhibition params dict; when given, training uses it as
        "nse_alt_train" instead of the group's default inhibition.
"""
import argparse
import copy
import json

PASS = {"type": "random_index", "strength": 5.0, "ri_variant": "threshold_linear",
        "ri_reach": 0.0, "ri_strength": 0.0, "ri_seed": 0, "ri_seed_from_time": False}


def derive(cfg, mode, seed_offset=1000, ds_override=None, train_inh=None):
    c = copy.deepcopy(cfg)
    ngm2 = c["ngm2"]
    groups = ngm2["neuron_groups"].values()
    inh = ngm2.setdefault("inhibitions", {})
    if train_inh is not None:
        inh["nse_alt_train"] = dict(PASS, **train_inh)
        for g in groups:
            g["inhibition_cfg"] = "nse_alt_train"
    if mode == "train":
        return c
    inh["nse_alt_pass"] = PASS
    for g in groups:
        g["frozen"] = True
        g["inhibition_cfg"] = {"probe": "nse_alt_pass", "inh": "nse_alt_train"}[mode]
    for ds in c["datasets"].values():
        for entry in ds.values():
            if "random_seed" in entry:
                entry["random_seed"] = int(entry["random_seed"]) + seed_offset
            entry.update(ds_override or {})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=["train", "probe", "inh"], default="probe")
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--override", default="{}", help="JSON merged into every dataset entry")
    ap.add_argument("--train-inh", default=None, help="JSON inhibition params used for training")
    a = ap.parse_args()
    with open(a.src) as f:
        cfg = json.load(f)
    out = derive(cfg, a.mode, a.seed_offset, json.loads(a.override),
                 json.loads(a.train_inh) if a.train_inh else None)
    with open(a.dst, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
