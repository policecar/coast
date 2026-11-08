#!/usr/bin/env python3
"""
ngm2 -> ngm2.5 config generator (design decision §13.3 obligation).

The ngm2.5 flat schema stores a group's structure inline (apical_cnt/proximal_cnt,
a flat `inputs` list with per-channel bank), whereas ngm2 is hierarchical:
  group -> neuron_sets -> dendrite_sets, wired by two layers of channel_mappings.
Rather than hand-maintain matched config pairs (an error source once already),
every ngm2.5 comparison config is DERIVED from its ngm2 original by this script.

Translation, per group:
  neuron_cnt              = the group's single neuron_set's neuron_count
  apical_cnt/proximal_cnt = sum of dendrite_count over the neuron's dendrite_sets,
                            bucketed by each set's dendrite type
  max_branch_level,       = the dendrite params (assumed uniform across the
  permanence_threshold,     group's dendrites; a divergence is reported)
  learning rates, mismatch
  branch_interval,        = the neuron params
  neuron_noise_range
  local_inhibition_strength, = the group params
  common_learning_rate,
  weight_filter, stochastic_win_thres, random_seed
  inputs = each group input channel traced group->neuron->dendrite to find the
           bank (dendrite type) it ultimately feeds.

The sim graph is copied with entity_type "ngm2_neuron_group" -> "ngm25_neuron_group".
Limitations (reported, not silently dropped): one neuron_set per group; uniform
dendrite params per group. Multi-set groups need a manual split.

Usage:  ngm2_to_ngm25_config.py IN.json OUT.json
"""
import json
import sys


def _trace_bank(group, neuron, dendrites, from_channel):
    """Follow from_channel: group channel_mapping -> neuron channel -> dendrite
    set -> dendrite cfg -> dendrite type. Returns 'apical' | 'proximal'."""
    # group mapping: from_channel -> (to_name = neuron set, to_channel = neuron ch)
    gm = next((m for m in group.get("channel_mappings", [])
               if m["from_channel"] == from_channel), None)
    if gm is None:
        # No explicit mapping: default to proximal (the main bank).
        return "proximal"
    neuron_channel = gm["to_channel"]
    # neuron mapping: neuron_channel -> (to_name = dendrite set, to_channel)
    nm = next((m for m in neuron.get("channel_mappings", [])
               if m["from_channel"] == neuron_channel), None)
    if nm is None:
        return "proximal"
    dset = next((d for d in neuron.get("dendrites", [])
                 if d["name"] == nm["to_name"]), None)
    if dset is None:
        return "proximal"
    dcfg = dendrites.get(dset["dendrite_cfg"], {})
    return dcfg.get("type", "proximal")


def _convert_group(name, group, neurons, dendrites, warnings):
    nsets = group.get("neurons", [])
    if len(nsets) != 1:
        warnings.append(f"group '{name}': expected exactly one neuron_set, got "
                        f"{len(nsets)} — using the first; split manually if needed")
    nset = nsets[0]
    neuron = neurons[nset["neuron_cfg"]]

    # Bucket dendrite counts by type; collect the (assumed uniform) dendrite cfg.
    apical_cnt = proximal_cnt = 0
    dcfgs = []
    for dset in neuron.get("dendrites", []):
        dcfg = dendrites[dset["dendrite_cfg"]]
        dcfgs.append((dset["dendrite_cfg"], dcfg))
        if dcfg.get("type", "proximal") == "apical":
            apical_cnt += dset["dendrite_count"]
        else:
            proximal_cnt += dset["dendrite_count"]

    # Group-level dendrite params come from a proximal dendrite (the main bank);
    # report if any two dendrite cfgs disagree on the shared scalars.
    shared = ("permanence_threshold", "max_branch_level",
              "default_primary_learning_rate", "default_secondary_learning_rate",
              "default_mismatch_act_thres", "default_mismatch_smoothing",
              "default_accumulated_theta_thres", "default_min_mismatch_deviation",
              "default_min_mismatch_percentage")
    base = next((c for _, c in dcfgs if c.get("type") == "proximal"), dcfgs[0][1])
    for cname, c in dcfgs:
        for k in shared:
            if c.get(k, base.get(k)) != base.get(k):
                warnings.append(f"group '{name}': dendrite '{cname}' field '{k}' "
                                f"differs from the flattened value (flat schema "
                                f"keeps ONE set); took the proximal/base value")

    # Trace each group input channel to its bank.
    inputs = []
    for gm in group.get("channel_mappings", []):
        ch = gm["from_channel"]
        inputs.append({"channel_name": ch,
                       "bank": _trace_bank(group, neuron, dendrites, ch)})
    # Group input_channels not covered by a mapping default to proximal.
    for ch in group.get("input_channels", []):
        if not any(i["channel_name"] == ch for i in inputs):
            inputs.append({"channel_name": ch, "bank": "proximal"})

    def g(d, k, default):
        return d.get(k, default)

    return {
        "neuron_cnt":   nset["neuron_count"],
        "apical_cnt":   apical_cnt,
        "proximal_cnt": proximal_cnt,
        "max_branch_level": g(base, "max_branch_level", 3),
        "permanence_threshold":      g(base, "permanence_threshold", 0.3),
        "local_inhibition_strength": g(group, "default_local_inhibition_strength", 10.0),
        "stochastic_win_thres":      g(group, "default_stochastic_win_thres", 0.8),
        "weight_filter": g(group, "default_weight_filter",
                           {"steepness": 0.5, "transition_point": 0.33}),
        "common_learning_rate": g(group, "default_common_learning_rate", 0.0001),
        "primary_learning_rate":   g(base, "default_primary_learning_rate", 0.01),
        "secondary_learning_rate": g(base, "default_secondary_learning_rate", 0.0001),
        "mismatch_act_thres":      g(base, "default_mismatch_act_thres", 0.8),
        "mismatch_smoothing":      g(base, "default_mismatch_smoothing", 0.001),
        "branch_interval":         g(neuron, "default_branch_interval", 5000),
        "min_mismatch_deviation":  g(base, "default_min_mismatch_deviation", 1.0),
        "accumulated_theta_thres": g(base, "default_accumulated_theta_thres", 2.0),
        "min_mismatch_percentage": g(base, "default_min_mismatch_percentage", 0.002),
        "neuron_noise_range":      g(neuron, "default_neuron_noise_range", [0.01, 0.05]),
        "penalty_policy": "stochastic",           # ngm2 semantics
        "inputs": inputs,
        "random_seed": g(group, "random_seed", 0),
        "inhibition_cfg": "",                      # synthesize max_softmax (beta = strength)
    }


def convert(cfg):
    warnings = []
    src = cfg.get("ngm2", {})
    dendrites = src.get("dendrites", {})
    neurons   = src.get("neurons", {})
    groups    = src.get("neuron_groups", {})

    ngm25_groups = {name: _convert_group(name, g, neurons, dendrites, warnings)
                    for name, g in groups.items()}

    out = {}
    if "datasets" in cfg:
        out["datasets"] = cfg["datasets"]
    out["ngm25"] = {"neuron_groups": ngm25_groups,
                    "inhibitions": src.get("inhibitions", {})}

    # Rewrite the sim graph: ngm2 group nodes become ngm25 group nodes.
    sim = json.loads(json.dumps(cfg.get("sim", {"nodes": [], "node_sets": []})))
    for node in sim.get("nodes", []):
        if node.get("entity_type") == "ngm2_neuron_group":
            node["entity_type"] = "ngm25_neuron_group"
    out["sim"] = sim
    return out, warnings


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} IN.json OUT.json")
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    out, warnings = convert(cfg)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=4)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"wrote {sys.argv[2]} "
          f"({len(out['ngm25']['neuron_groups'])} ngm2.5 group(s))")


if __name__ == "__main__":
    main()
