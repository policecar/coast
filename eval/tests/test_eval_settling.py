"""`scripts/eval_settling.py`'s reference windows follow the template's hop count (#276): the
pixel template keeps the registered steps 3..15, a retina template starts one step later."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_settling.py"
TEMPLATE = Path(__file__).parents[1] / "configs" / "mnist_ngm25_frozen_eval.json"


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("eval_settling", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


es = load_script()


def with_retina(cfg: dict[str, Any]) -> dict[str, Any]:
    """The pixel template with a node between the dataset and the group: two hops."""
    nodes = cfg["sim"]["nodes"]
    nodes.insert(1, {"id": 10, "entity_type": "retina", "inputs": [[0, "in"]]})
    group = next(n for n in nodes if n["entity_type"].endswith("neuron_group"))
    group["inputs"] = [[10, "on"]]
    return cfg


def test_pixel_template_keeps_the_registered_windows() -> None:
    w = es.windows(json.loads(TEMPLATE.read_text()))
    assert (w.ref, w.full, w.lead) == ((3, 15), (3, 30), 2)
    assert w.tau_steps == (1, 2, 3)
    assert w.tau_windows == {"avg5": (3, 5), "avg10": (3, 10), "ref": (3, 15), "ref30": (3, 30)}


def test_a_retina_template_starts_one_step_later() -> None:
    w = es.windows(with_retina(json.loads(TEMPLATE.read_text())))
    assert (w.ref, w.full, w.lead) == ((4, 15), (4, 30), 3)
    assert w.tau_steps == (1, 2, 3, 4)
    assert w.tau_windows == {"avg5": (4, 5), "avg10": (4, 10), "ref": (4, 15), "ref30": (4, 30)}
