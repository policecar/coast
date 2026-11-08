"""The three milestone-1 templates: one model, three front-ends.

The baseline feeds MNIST pixels to the group; the retina templates put an ON and an OFF
retina node in between, without and with divisive normalisation. Everything outside the
front-end must be the same in all three, so an A/B between them compares front-ends only.

The divnorm template's norm_sigma and gain come from scripts/retina_norm_sigma_probe.py,
calibrated on test images 9000..9999 (mnist_retina_norm_sigma.json): an evaluation with it
must use test indices below 9000, and milestone 1 draws its images from the indices below
protocol.SAMPLE_POOL.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from coast_eval import protocol as pr

CONFIGS = Path(__file__).parents[1] / "configs"
BASELINE = CONFIGS / "mnist_ngm25_frozen_eval.json"
RETINA = CONFIGS / "mnist_ngm25_frozen_eval_retina.json"
DIVNORM = CONFIGS / "mnist_ngm25_frozen_eval_retina_divnorm.json"
NORM_SIGMA = CONFIGS / "mnist_retina_norm_sigma.json"
MNIST_SHAPE = (28, 28)  # rows, cols of an MNIST / MNIST-C image: one mnist_input output


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def output_width(cfg: dict[str, Any], node_id: int) -> int:
    """Output size of a source node, as coast sizes it; a retina's input must match the
    output of the node it reads (retina_io::init_io refuses it otherwise)."""
    nodes = {n["id"]: n for n in cfg["sim"]["nodes"]}
    node = nodes[node_id]
    if node["entity_type"] == "mnist_input":
        return math.prod(MNIST_SHAPE)
    if node["entity_type"] == "retina":
        r = cfg["retina"][node["entity_cfg"]]
        ((src, _),) = node["inputs"]
        if r["width"] * r["height"] * r["channels"] != output_width(cfg, src):
            raise ValueError(f"retina {node_id}: input size differs from node {src}'s output")
        return r["width"] * r["height"] * (1 if r["luminance"] else r["channels"])
    raise ValueError(f"node {node_id}: no width rule for {node['entity_type']}")


def proximal_width(cfg: dict[str, Any]) -> int:
    """The group's proximal input width: the sum of its routed channels' sizes, as
    ngm25's init_io concatenates them. The routes must name exactly the wired channels."""
    (group,) = (n for n in cfg["sim"]["nodes"] if n["entity_type"] == "ngm25_neuron_group")
    routes = cfg["ngm25"]["neuron_groups"][group["entity_cfg"]]["inputs"]
    wired = {channel: src for src, channel in group["inputs"]}
    if sorted(wired) != sorted(r["channel_name"] for r in routes):
        raise ValueError(f"routes {routes} do not match the wiring {group['inputs']}")
    return sum(
        output_width(cfg, wired[r["channel_name"]]) for r in routes if r["bank"] == "proximal"
    )


def without_front_end(cfg: dict[str, Any]) -> dict[str, Any]:
    """The template minus everything a front-end may change: the retina section and nodes,
    and the group's input routes and wiring."""
    c = copy.deepcopy(cfg)
    c.pop("retina", None)
    c["sim"]["nodes"] = [n for n in c["sim"]["nodes"] if n["entity_type"] != "retina"]
    for node in c["sim"]["nodes"]:
        if node["entity_type"] == "ngm25_neuron_group":
            del node["inputs"]
            del c["ngm25"]["neuron_groups"][node["entity_cfg"]]["inputs"]
    return c


@pytest.mark.parametrize(("path", "planes"), [(BASELINE, 1), (RETINA, 2), (DIVNORM, 2)])
def test_proximal_width_is_the_planes(path: Path, planes: int) -> None:
    cfg = load(path)
    retinas = [n for n in cfg["sim"]["nodes"] if n["entity_type"] == "retina"]
    assert len(retinas) == (planes if planes > 1 else 0)
    assert proximal_width(cfg) == planes * math.prod(MNIST_SHAPE)


def test_width_check_sees_a_dropped_plane_and_a_wrong_retina_size() -> None:
    cfg = load(RETINA)
    group = cfg["ngm25"]["neuron_groups"]["L4-mnist"]
    one_plane = copy.deepcopy(cfg)
    one_plane["ngm25"]["neuron_groups"]["L4-mnist"]["inputs"] = group["inputs"][:1]
    with pytest.raises(ValueError, match="do not match the wiring"):
        proximal_width(one_plane)
    one_plane["sim"]["nodes"][-1]["inputs"] = one_plane["sim"]["nodes"][-1]["inputs"][:1]
    assert proximal_width(one_plane) == math.prod(MNIST_SHAPE)
    narrow = copy.deepcopy(cfg)
    narrow["retina"]["off"]["width"] = 27
    with pytest.raises(ValueError, match="retina 11: input size"):
        proximal_width(narrow)


def test_templates_differ_only_in_the_front_end() -> None:
    base = without_front_end(load(BASELINE))
    group = base["ngm25"]["neuron_groups"]["L4-mnist"]
    assert (group["neuron_cnt"], group["inhibition_cfg"]) == (50, "ab_pinned")
    for path in (RETINA, DIVNORM):
        assert without_front_end(load(path)) == base


def test_front_end_comparison_sees_a_model_drift() -> None:
    drifted = load(RETINA)
    drifted["ngm25"]["neuron_groups"]["L4-mnist"]["primary_learning_rate"] *= 2
    assert without_front_end(drifted) != without_front_end(load(BASELINE))


def check_planes(linear: dict[str, Any], divnorm: dict[str, Any], measured: dict[str, Any]) -> None:
    """B is A plus normalisation at the measured norm_sigma and the matched gain: the gain
    at which a median clean digit has A's output RMS. A runs at the gain the probe used."""
    clipping = measured["clean_clipping"]
    changed = ("normalize", "norm_sigma", "gain")
    for pol in ("on", "off"):
        assert linear[pol]["polarity"] == divnorm[pol]["polarity"] == pol
        assert linear[pol]["normalize"] is False and "norm_sigma" not in linear[pol]
        assert linear[pol]["gain"] == clipping["linear_at_template_gain"]["gain"]
        assert divnorm[pol]["normalize"] is True
        assert divnorm[pol]["norm_sigma"] == measured["norm_sigma"]
        assert divnorm[pol]["gain"] == clipping["normalized_at_matched_gain"]["gain"]
        rest = {k: v for k, v in divnorm[pol].items() if k not in changed}
        assert rest == {k: v for k, v in linear[pol].items() if k not in changed}
    # the planes differ in polarity and noise seed only: the probe's one x describes both
    for planes in (linear, divnorm):
        on, off = ({k: v for k, v in planes[p].items() if k not in ("polarity", "random_seed")}
                   for p in ("on", "off"))  # fmt: skip
        assert on == off
        assert planes["on"]["random_seed"] != planes["off"]["random_seed"]


def test_retina_planes() -> None:
    measured = load(NORM_SIGMA)
    assert 0.0 < measured["norm_sigma"] < 1.0
    first, last = measured["test_indices"]  # the calibration images: none below SAMPLE_POOL
    assert pr.SAMPLE_POOL <= first <= last < 10_000
    check_planes(load(RETINA)["retina"], load(DIVNORM)["retina"], measured)


@pytest.mark.parametrize(
    ("key", "value"), [("gain", 2.0), ("norm_sigma", 0.05), ("center_sigma", 1.5)]
)
def test_plane_check_sees_a_drift(key: str, value: float) -> None:
    divnorm = load(DIVNORM)["retina"]
    for plane in divnorm.values():  # both planes, so only the tie to A or the probe can fail
        plane[key] = value
    with pytest.raises(AssertionError):
        check_planes(load(RETINA)["retina"], divnorm, load(NORM_SIGMA))
