"""Matrix expansion: what each job's config says, and the alignment precondition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from coast_eval import datasets as ds
from coast_eval import protocol as pr

TEMPLATE = Path(__file__).parents[1] / "configs" / "mnist_ngm25_frozen_eval.json"
# the same protocol on the retina front-ends: (template, ids of its nodes)
TEMPLATES = [
    (TEMPLATE, [0, 1]),
    (TEMPLATE.with_name("mnist_ngm25_frozen_eval_retina.json"), [0, 10, 11, 1]),
    (TEMPLATE.with_name("mnist_ngm25_frozen_eval_retina_divnorm.json"), [0, 10, 11, 1]),
]
TRAIN, TEST = list(range(12)), [7, 3, 11, 0]  # explicit, unsorted order
PINNED_SAMPLE_HEAD, PINNED_SAMPLE_LAST = [9, 14, 20, 21], 8984
PINNED_TUNING_HEAD = [42393, 50201, 51168, 29760]


@pytest.fixture
def inputs(tmp_path: Path) -> pr.Inputs:
    root = tmp_path / "data"
    (root / "mnist").mkdir(parents=True)
    ds.write_idx(root / ds.TRAIN_LABELS, np.zeros(12, dtype=np.uint8))
    clean = np.full((20, 28, 28), 9, dtype=np.uint8)
    ds.write_idx(root / ds.TEST_IMAGES, clean)
    return pr.write_inputs(tmp_path / "inputs", root, TRAIN, TEST)


def test_evaluation_sample_is_fixed_sorted_and_clear_of_the_calibration_images() -> None:
    """Every comparison must read the same images, so the values are pinned: a numpy that
    changed the generator's stream would fail here, not move a result without a trace."""
    sample = pr.evaluation_sample(1000)
    assert sample[:4] == PINNED_SAMPLE_HEAD and sample[-1] == PINNED_SAMPLE_LAST
    assert sample == sorted(set(sample)) and len(sample) == 1000
    assert sample[-1] < pr.SAMPLE_POOL == 9000
    assert len(pr.evaluation_sample(pr.SAMPLE_POOL)) == pr.SAMPLE_POOL
    for n in (0, pr.SAMPLE_POOL + 1):
        with pytest.raises(ValueError, match="test images"):
            pr.evaluation_sample(n)


def template(path: Path = TEMPLATE) -> dict[str, Any]:
    return json.loads(path.read_text())


def cfg(job: pr.Job) -> dict[str, Any]:
    return json.loads(job.config_text)


def dataset(job: pr.Job) -> dict[str, Any]:
    return cfg(job)["datasets"]["mnist"]["default"]


def test_write_inputs(inputs: pr.Inputs) -> None:
    assert Path(inputs.test_manifest).read_text() == "7\n3\n11\n0\n"
    assert Path(inputs.train_manifest).read_text().split() == [str(i) for i in TRAIN]
    zeros = ds.read_idx_images(Path(inputs.zeros_images))
    assert zeros.shape == (20, 28, 28)  # same count as the clean split: ids keep their meaning
    assert not zeros.any()


@pytest.mark.parametrize(
    ("train", "test"),
    [([0, 1, 1], TEST), (TRAIN, [3, 3]), ([12], TEST), (TRAIN, [20]), (TRAIN, [-1]), ([], TEST)],
)
def test_write_inputs_refuses_bad_indices(
    tmp_path: Path, inputs: pr.Inputs, train: list[int], test: list[int]
) -> None:
    with pytest.raises(ValueError, match="unique and in"):
        pr.write_inputs(tmp_path / "again", tmp_path / "data", train, test)


@pytest.mark.parametrize(("path", "node_ids"), TEMPLATES)
def test_train_jobs(inputs: pr.Inputs, path: Path, node_ids: list[int]) -> None:
    tpl = template(path)
    jobs = pr.train_jobs(
        tpl, inputs, data_ref="./data", seeds=[0, 4, 2], steps=50_000, change_interval=5
    )
    assert [(j.seed, j.condition, j.steps) for j in jobs] == [
        (0, "train", 50_000),
        (4, "train", 50_000),
        (2, "train", 50_000),
    ]
    for job in jobs:
        d = dataset(job)
        assert d["image_file"] == "./data/mnist/train-images-idx3-ubyte"
        assert d["label_file"] == "./data/mnist/train-labels-idx1-ubyte"
        assert d["index_manifest"] == inputs.train_manifest
        assert (d["change_interval"], d["noise_range"], d["shuffle"]) == (5, [0.0, 0.0], True)
        assert pr.steps_per_image(cfg(job)) == 5
        c = cfg(job)
        assert c["frozen"] is False  # the run's (SPEC.md S.5.14)
        assert "frozen" not in c["ngm25"]["neuron_groups"]["L4-mnist"]
        assert "restore_sequence_from_checkpoint" not in d
        assert c["observe"]["groups"] == {}
        nodes = c["sim"]["nodes"]
        assert [n["id"] for n in nodes] == node_ids  # the seed reaches every node
        assert [n["random_seed_offset"] for n in nodes] == [job.seed] * len(node_ids)
        # the retina's noise floor is its null signal, like the neurons' baseline noise:
        # the protocol leaves the retina section as the template has it
        assert c.get("retina") == tpl.get("retina")
        # baseline neuron noise stays at the model default
        assert c["ngm25"]["neuron_groups"]["L4-mnist"]["neuron_noise_range"] == [0.01, 0.05]
        # the #243 inhibition: the template is that configuration, not a default to tidy
        group = c["ngm25"]["neuron_groups"]["L4-mnist"]
        assert (group["local_inhibition_strength"], group["inhibition_cfg"]) == (10.0, "ab_pinned")
        assert c["ngm25"]["inhibitions"] == {"ab_pinned": {"type": "max_softmax", "strength": 10.0}}
    with pytest.raises(ValueError, match="seeds"):
        pr.train_jobs(tpl, inputs, data_ref="d", seeds=[1, 1], steps=10, change_interval=5)


@pytest.mark.parametrize(("path", "node_ids"), TEMPLATES)
@pytest.mark.parametrize("base_step", [0, 2000])
def test_frozen_jobs_cover_every_condition(
    inputs: pr.Inputs, base_step: int, path: Path, node_ids: list[int]
) -> None:
    tpl = template(path)
    jobs = pr.frozen_jobs(
        tpl, inputs, data_ref="./data", seed=3, conditions=pr.CONDITIONS, base_step=base_step
    )
    assert len(pr.CONDITIONS) == 17
    assert [j.condition for j in jobs] == list(pr.CONDITIONS)
    images = {j.condition: dataset(j)["image_file"] for j in jobs}
    assert len(set(images.values())) == 17
    assert images["clean"] == "./data/mnist/t10k-images-idx3-ubyte"
    assert images["fog"] == "./data/mnist_c/fog/t10k-images-idx3-ubyte"
    assert images["zeros"] == inputs.zeros_images
    for job in jobs:
        assert (job.seed, job.steps) == (3, 60)
        d = dataset(job)
        assert d["index_manifest"] == inputs.test_manifest
        assert (d["change_interval"], d["noise_range"], d["shuffle"]) == (15, [0.0, 0.0], False)
        assert pr.steps_per_image(cfg(job)) == pr.FROZEN_STEPS_PER_IMAGE
        # a frozen run: coast starts the sequence fresh from the trained state, and
        # continues it from the trial's own checkpoint
        c = cfg(job)
        assert c["frozen"] is True
        assert "frozen" not in c["ngm25"]["neuron_groups"]["L4-mnist"]
        assert "restore_sequence_from_checkpoint" not in d
        window = {"ranges": [[base_step, base_step + 59]], "combine": "or"}
        assert c["observe"]["groups"] == {"population": window, "input": window}
        assert [n["id"] for n in c["sim"]["nodes"]] == node_ids
        assert [n["random_seed_offset"] for n in c["sim"]["nodes"]] == [3] * len(node_ids)
        assert c.get("retina") == tpl.get("retina")
    labels = {j.condition: dataset(j)["label_file"] for j in jobs}
    assert labels["clean"] == labels["zeros"] == "./data/mnist/t10k-labels-idx1-ubyte"
    assert labels["fog"] == "./data/mnist_c/fog/t10k-labels-idx1-ubyte"


def test_frozen_jobs_refuse_unknown_or_repeated_conditions(inputs: pr.Inputs) -> None:
    tpl = template()
    with pytest.raises(ValueError, match="unknown MNIST-C"):
        pr.frozen_jobs(tpl, inputs, data_ref="d", seed=0, conditions=["blur"], base_step=0)
    with pytest.raises(ValueError, match="conditions"):
        pr.frozen_jobs(tpl, inputs, data_ref="d", seed=0, conditions=["fog", "fog"], base_step=0)


def test_overrides_never_add_keys(inputs: pr.Inputs) -> None:
    tpl = template()
    del tpl["datasets"]["mnist"]["default"]["shuffle"]
    with pytest.raises(KeyError, match="shuffle"):
        pr.train_jobs(tpl, inputs, data_ref="d", seeds=[0], steps=10, change_interval=5)


def test_check_aligned() -> None:
    pr.check_aligned(np.array(TEST, dtype=np.int64), TEST)
    for observed, at in [([7, 11, 3, 0], 1), ([7, 3, 11], 3), ([7, 3, 11, 0, 0], 4)]:
        with pytest.raises(ValueError, match=f"first difference at position {at}"):
            pr.check_aligned(np.array(observed, dtype=np.int64), TEST)
    with pytest.raises(ValueError, match="expected ids"):
        pr.check_aligned(np.array([1, 1], dtype=np.int64), [1, 1])


def with_retina(tpl: dict[str, Any]) -> dict[str, Any]:
    """The #258 retina template's node graph: mnist -> ON and OFF retinas -> group."""
    nodes = tpl["sim"]["nodes"]
    group = next(n for n in nodes if n["entity_type"] == "ngm25_neuron_group")
    retinas = [
        {"id": 10, "entity_type": "retina", "inputs": [[0, "in"]]},
        {"id": 11, "entity_type": "retina", "inputs": [[0, "in"]]},
    ]
    group["inputs"] = [[10, "on"], [11, "off"]]
    nodes[1:1] = retinas
    return tpl


def test_settle_lead_counts_hops_from_the_dataset() -> None:
    assert pr.settle_lead(template()) == 1  # pixels -> group
    assert pr.settle_lead(with_retina(template())) == 2
    # The longest path decides: a direct pixel route beside a two-retina chain.
    tpl = with_retina(template())
    tpl["sim"]["nodes"][1]["inputs"] = [[11, "in"]]  # 0 -> 11 -> 10 -> group
    tpl["sim"]["nodes"][-1]["inputs"].append([0, "ga"])
    assert pr.settle_lead(tpl) == 3


def test_settle_lead_refuses_graphs_without_one_reachable_group() -> None:
    tpl = template()
    tpl["sim"]["nodes"][0]["inputs"] = [[1, "loop"]]
    with pytest.raises(ValueError, match="cycle"):
        pr.settle_lead(tpl)
    tpl = template()
    tpl["sim"]["nodes"][1]["inputs"] = []
    with pytest.raises(ValueError, match="no inputs"):
        pr.settle_lead(tpl)
    tpl = template()
    tpl["sim"]["nodes"][1]["inputs"] = [[7, "ga"]]
    with pytest.raises(ValueError, match="unknown node 7"):
        pr.settle_lead(tpl)
    tpl = template()
    tpl["sim"]["nodes"].append(dict(tpl["sim"]["nodes"][1], id=2))
    with pytest.raises(ValueError, match="exactly one"):
        pr.settle_lead(tpl)


# --- the tuning split of a sweep --------------------------------------------------------


def test_tuning_split_is_fixed_and_a_prefix_of_its_pool() -> None:
    """Pinned like the evaluation sample; every n reads the first rows of the tuning files,
    which hold the pool's images in this order."""
    pool = pr.tuning_split(pr.TUNING_POOL)
    assert pool[:4] == PINNED_TUNING_HEAD
    assert len(set(pool)) == pr.TUNING_POOL and 0 <= min(pool) and max(pool) < pr.TRAIN_COUNT
    assert pr.tuning_split(7) == pool[:7]
    for n in (0, pr.TUNING_POOL + 1):
        with pytest.raises(ValueError, match="tuning images"):
            pr.tuning_split(n)


@pytest.fixture
def tuning(tmp_path: Path) -> tuple[Path, pr.TuningInputs]:
    root = tmp_path / "data"
    (root / "mnist").mkdir(parents=True)
    rows = np.arange(pr.TRAIN_COUNT)
    ds.write_idx(root / ds.TRAIN_IMAGES, np.stack([rows % 256, rows // 256], 1)
                 .astype(np.uint8).reshape(-1, 2, 1))  # fmt: skip
    ds.write_idx(root / ds.TRAIN_LABELS, (rows % 10).astype(np.uint8))
    return root, pr.write_tuning_inputs(tmp_path / "inputs", root, 5)


def test_write_tuning_inputs_holds_the_tuning_images_out_of_training(
    tuning: tuple[Path, pr.TuningInputs],
) -> None:
    root, t = tuning
    assert t.tuning_indices == tuple(pr.tuning_split(5))
    train = [int(i) for i in Path(t.train_manifest).read_text().split()]
    assert len(train) == pr.TRAIN_COUNT - 5 and train == sorted(train)
    assert not set(train) & set(t.tuning_indices)  # disjoint, and nothing else left out
    assert set(train) | set(t.tuning_indices) == set(range(pr.TRAIN_COUNT))
    assert Path(t.tuning_manifest).read_text() == "0\n1\n2\n3\n4\n"  # rows of the files
    images = ds.read_idx_images(Path(t.images))
    pool = pr.tuning_split(pr.TUNING_POOL)
    assert images.shape == (pr.TUNING_POOL, 2, 1)  # the whole pool, as MNIST-C's files
    train_images = ds.read_idx_images(root / ds.TRAIN_IMAGES)
    assert np.array_equal(images, train_images[pool])  # row r is train image pool[r]
    assert np.array_equal(ds.read_idx_labels(Path(t.labels)), np.asarray(pool) % 10)
    zeros = ds.read_idx_images(Path(t.zeros_images))
    assert zeros.shape == images.shape and not zeros.any()


def test_write_tuning_inputs_refuses_a_split_the_pinned_files_were_not_written_for(
    tuning: tuple[Path, pr.TuningInputs], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MNIST-C tuning files hold the rows of one permutation; another (a numpy whose
    Generator stream changed, here another seed) would pair their rows with other clean
    images and labels and train on the images they hold. The real one passes (`tuning`)."""
    root, _ = tuning
    monkeypatch.setattr(pr, "TUNING_SEED", pr.TUNING_SEED + 1)
    with pytest.raises(ValueError, match="pinned MNIST-C tuning files"):
        pr.write_tuning_inputs(tmp_path / "other", root, 5)
    assert not (tmp_path / "other").exists()  # refused before anything is written


def test_tuning_jobs_read_the_tuning_files_and_never_the_test_split(
    tuning: tuple[Path, pr.TuningInputs],
) -> None:
    _, t = tuning
    tpl = template()
    jobs = pr.tuning_jobs(tpl, t, data_ref="./data", seed=2, conditions=pr.CONDITIONS,
                          base_step=500)  # fmt: skip
    assert [j.condition for j in jobs] == list(pr.CONDITIONS)
    for job in jobs:
        d = dataset(job)
        assert job.steps == 5 * pr.FROZEN_STEPS_PER_IMAGE
        assert (d["index_manifest"], d["label_file"]) == (t.tuning_manifest, t.labels)
        assert "t10k" not in d["image_file"] and cfg(job)["frozen"] is True
        window = {"ranges": [[500, 500 + 74]], "combine": "or"}
        assert cfg(job)["observe"]["groups"] == {"population": window, "input": window}
    images = {j.condition: dataset(j)["image_file"] for j in jobs}
    assert (images["clean"], images["zeros"]) == (t.images, t.zeros_images)
    assert images["fog"] == "./data/mnist_c/fog/tuning-images-idx3-ubyte"
    (train,) = pr.train_jobs(tpl, t, data_ref="./data", seeds=[2], steps=10, change_interval=1)
    assert dataset(train)["index_manifest"] == t.train_manifest
    for bad in (f"./data/{ds.TEST_IMAGES}", "./data/mnist_c/fog/t10k-images-idx3-ubyte"):
        wrong = msgspec.structs.replace(t, images=bad)
        with pytest.raises(ValueError, match="must not read the test split"):
            pr.tuning_jobs(tpl, wrong, data_ref="./data", seed=2, conditions=["clean"],
                           base_step=0)  # fmt: skip
    labelled = msgspec.structs.replace(t, labels=f"./data/{ds.TEST_LABELS}")
    with pytest.raises(ValueError, match="must not read the test split"):
        pr.tuning_jobs(tpl, labelled, data_ref="./data", seed=2, conditions=["zeros"],
                       base_step=0)  # fmt: skip
