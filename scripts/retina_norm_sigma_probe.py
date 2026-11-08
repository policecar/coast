#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "coast-eval",
#   "h5py",
#   "numpy",
# ]
#
# [tool.uv.sources]
# coast-eval = { path = "../eval", editable = true }
# ///
"""Measure the retina's gain-free response x on MNIST and MNIST-C and choose norm_sigma.

Usage:
    scripts/retina_norm_sigma_probe.py --coast build-headless/coast --data-root data \
        --template eval/configs/mnist_ngm25_frozen_eval_retina.json \
        --work-dir tmp/norm_sigma --out eval/configs/mnist_retina_norm_sigma.json

The retina divides x by norm_sigma + RMS(x), the RMS over the whole plane
(src/retina/retina.cpp). x is the signed centre-surround response before the ON/OFF split,
so the ON and the OFF node divide by the same RMS(x). Its noise floor is added after
polarisation and never enters x; the only noise that can is the dataset's input noise.

x comes from coast itself: each condition runs the template with the retina at gain 1,
normalisation off and a 1e-9 noise floor, logs every io buffer (--log-buffers), and takes
x = ON - OFF. Two checks must pass before a number is used: x matches a numpy
re-implementation of the filter applied to the logged dataset frame of the step before
(one buffered hop), and a run with normalisation on at the chosen norm_sigma and the
template's gain matches clip(gain * x / (norm_sigma + RMS(x)), -1, 1).

Conditions, on the test images first..first+count-1: clean; each MNIST-C corruption; an
all-zero image without input noise (checked to be exactly 0, not reported) and with input
noise drawn from the template's retina noise_range (the noise-only plane).

Calibration split: MNIST-C exists for the test split only, so the calibration images are a
range of the test split kept apart from evaluation, by default 9000..9999 (--first
protocol.SAMPLE_POOL, --count 1000). An evaluation that uses the measured norm_sigma must use
test indices below --first (milestone 1 draws from the indices below protocol.SAMPLE_POOL);
the result JSON records the range.

Rule (agreed with the owner): the knee sits above the noise-only plane and well below a
digit plane; norm_sigma is the geometric mean of the two medians, rounded to two
significant figures.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from coast_eval import datasets as ds
from coast_eval import protocol as pr

_FILTER_KEYS = (
    "width", "height", "channels", "luminance", "center_sigma", "surround_sigma",
    "surround_weight",
)  # fmt: skip
_TOL = 1e-5  # float32 blur sums vs float64 numpy
_FLOOR = 1e-9  # the probe's retina noise_range upper end: validate() refuses 0


def gaussian_kernel(sigma: float) -> NDArray[np.float64]:
    r = math.ceil(3.0 * sigma)
    i = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * i * i / (sigma * sigma))
    return k / k.sum()


def blur(plane: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Separable blur of (n, h, w) planes with clamp-to-edge borders, as retina.cpp."""
    r = kernel.size // 2
    h, w = int(plane.shape[1]), int(plane.shape[2])

    def taps(n: int) -> NDArray[np.intp]:  # (n, 2r+1) clamped source indices
        return np.clip(np.add.outer(np.arange(n), np.arange(-r, r + 1)), 0, n - 1)

    tmp = (plane[:, :, taps(w)] * kernel).sum(axis=-1)
    return np.einsum("nikj,k->nij", tmp[:, taps(h), :], kernel)


def dog(images: NDArray[np.float64], p: dict[str, Any]) -> NDArray[np.float64]:
    """Gain-free signed response x of (n, h, w) single-channel planes."""
    k = float(p["surround_weight"])
    mean = images.mean(axis=(1, 2), keepdims=True)
    centre = blur(images, gaussian_kernel(float(p["center_sigma"])))
    surround = blur(images, gaussian_kernel(float(p["surround_sigma"])))
    return centre - k * surround - (1.0 - k) * mean


def shared_filter(template: dict[str, Any]) -> dict[str, Any]:
    """The filter parameters of the template's ON and OFF planes, which must agree (else the
    planes have different x and this probe's single x does not describe them)."""
    planes = template["retina"]
    if sorted(planes) != ["off", "on"]:
        raise ValueError(f"template retina section must be exactly on/off, got {sorted(planes)}")
    on, off = ({key: planes[pol][key] for key in _FILTER_KEYS} for pol in ("on", "off"))
    if on != off:
        raise ValueError(f"ON and OFF filter parameters differ: {on} vs {off}")
    if on["channels"] != 1 or on["luminance"]:
        raise ValueError("this probe handles single-channel planes only")
    return on


def run(
    coast: Path,
    template: dict[str, Any],
    work: Path,
    *,
    images: Path,
    labels: Path,
    manifest: Path,
    steps: int,
    input_noise: list[float],
    retina: dict[str, Any],
) -> h5py.File:
    """Run the template for `steps` steps with the given retina overrides on both planes;
    return the open buffer log. `coast` is resolved before coast runs in `work`."""
    cfg: dict[str, Any] = json.loads(json.dumps(template))
    cfg["datasets"]["mnist"]["default"].update(
        image_file=str(images.resolve()),
        label_file=str(labels.resolve()),
        index_manifest=str(manifest.resolve()),
        change_interval=1,
        noise_range=input_noise,
    )
    for plane in cfg["retina"].values():
        plane.update(retina)
    cfg["observe"]["groups"] = {}
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps(cfg, indent=2))
    log = work.resolve() / "buffers.h5"
    log.unlink(missing_ok=True)
    subprocess.run(
        [str(coast.resolve()), "-H", "-n", str(steps), "--log-buffers", str(log), "config.json"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    return h5py.File(log, "r")


def frames(log: h5py.File, node_id: int) -> NDArray[np.float64]:
    data = log[f"steps/buffers/{node_id}/data"]
    if not isinstance(data, h5py.Dataset):
        raise ValueError(f"buffer log has no dataset for node {node_id}")
    return np.asarray(data[()], dtype=np.float64)


def node_ids(template: dict[str, Any]) -> tuple[int, int, int]:
    """(dataset, ON retina, OFF retina) node ids; coast names each buffer by its node id."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for n in template["sim"]["nodes"]:
        by_type.setdefault(n["entity_type"], []).append(n)
    (data,) = by_type["mnist_input"]
    ret = {n["entity_cfg"]: n["id"] for n in by_type["retina"]}
    return int(data["id"]), int(ret["on"]), int(ret["off"])


def measure(
    log: h5py.File, ids: tuple[int, int, int], p: dict[str, Any], count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(x, input) of the first `count` presentations, each (count, h, w); x checked against
    the numpy filter applied to the input."""
    h, w = int(p["height"]), int(p["width"])
    inp = frames(log, ids[0]).reshape(-1, h, w)
    x = (frames(log, ids[1]) - frames(log, ids[2])).reshape(-1, h, w)
    # retina frame t reads the dataset frame t - 1 (one buffered hop)
    x, inp = x[1 : count + 1], inp[:count]
    err = np.abs(x - dog(inp, p)).max()
    if err > _TOL:
        raise ValueError(f"coast x differs from the numpy filter by {err:.2e} > {_TOL}")
    return x, inp


def rms(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt((x * x).mean(axis=(1, 2)))


def stats(x: NDArray[np.float64]) -> dict[str, float]:
    q = np.percentile(rms(x), [5, 50, 95])
    return {"p5": float(q[0]), "median": float(q[1]), "p95": float(q[2])}


def rounded(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {k: rounded(e) for k, e in v.items()}  # pyright: ignore[reportUnknownVariableType]
    return v


def round_sig(v: float, digits: int) -> float:
    return float(f"{v:.{digits - 1}e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--coast", type=Path, required=True, help="headless coast binary")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True, help="a retina eval template")
    ap.add_argument("--work-dir", type=Path, required=True, help="scratch (configs, logs)")
    ap.add_argument("--out", type=Path, required=True, help="result JSON (path-free)")
    ap.add_argument(
        "--first", type=int, default=pr.SAMPLE_POOL, help="first calibration test image"
    )
    ap.add_argument("--count", type=int, default=1000, help="test images first..first+count-1")
    args = ap.parse_args()
    if args.count < 500:
        raise SystemExit("--count must be >= 500")

    template = json.loads(args.template.read_text())
    p = shared_filter(template)
    ids = node_ids(template)
    root: Path = args.data_root
    ds.verify_pins(
        root,
        {k: v for k, v in ds.PINS.items() if k.startswith("mnist_c/") or "t10k" in k},
    )
    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    clean = ds.read_idx_images(root / ds.TEST_IMAGES)
    first, last = args.first, args.first + args.count - 1
    if first < 0 or last >= clean.shape[0]:
        raise SystemExit(f"test images {first}..{last} are not in [0, {clean.shape[0]})")
    manifest = work / "indices.txt"
    manifest.write_text("".join(f"{i}\n" for i in range(first, last + 1)))
    zeros = work / "zeros-images-idx3"
    ds.write_idx(zeros, np.zeros_like(clean))
    noise = [float(v) for v in template["retina"]["on"]["noise_range"]]
    probe = {"gain": 1.0, "normalize": False, "noise_range": [0.0, _FLOOR]}
    labels = root / ds.TEST_LABELS

    conditions: dict[str, tuple[Path, list[float]]] = {
        "clean": (root / ds.TEST_IMAGES, [0.0, 0.0]),
        "blank": (zeros, [0.0, 0.0]),
        "blank_input_noise": (zeros, noise),
    } | {c: (root / ds.corruption_files(c)[0], [0.0, 0.0]) for c in ds.MNIST_C}
    xs: dict[str, NDArray[np.float64]] = {}
    for name, (images, input_noise) in conditions.items():
        with run(
            args.coast, template, work / name, images=images, labels=labels,
            manifest=manifest, steps=args.count + 1, input_noise=input_noise, retina=probe,
        ) as log:  # fmt: skip
            xs[name], inp = measure(log, ids, p, args.count)
        if input_noise == [0.0, 0.0]:  # presentation i is image first + i, as loaded
            want = ds.read_idx_images(images)[first : last + 1] / 255.0
            if inp.shape != want.shape or np.abs(inp - want).max() > _TOL:
                raise ValueError(f"{name}: logged input frames are not images {first}..{last}")
    if np.abs(xs["blank"]).max() > _FLOOR:  # 0 up to the probe's own noise floor
        raise ValueError("the blank plane without input noise is not 0")
    # the blank plane is a self-check (above), not a result
    measured = {name: stats(x) for name, x in xs.items() if name != "blank"}
    floor, digit = measured["blank_input_noise"]["median"], measured["clean"]["median"]
    sigma = round_sig(math.sqrt(floor * digit), 2)

    # normalisation at the linear template's gain and the chosen sigma (the clip shares)
    gain = float(template["retina"]["on"]["gain"])
    norm = {"gain": gain, "normalize": True, "norm_sigma": sigma, "noise_range": [0.0, _FLOOR]}
    with run(
        args.coast, template, work / "clean_normalized", images=root / ds.TEST_IMAGES,
        labels=labels, manifest=manifest, steps=args.count + 1, input_noise=[0.0, 0.0],
        retina=norm,
    ) as log:  # fmt: skip
        h, w = int(p["height"]), int(p["width"])
        d = (frames(log, ids[1]) - frames(log, ids[2])).reshape(-1, h, w)[1 : args.count + 1]
    x = xs["clean"]
    want = np.clip(gain * x / (sigma + rms(x))[:, None, None], -1.0, 1.0)
    err = np.abs(d - want).max()
    if err > _TOL * gain / sigma:
        raise ValueError(f"normalised output differs from the formula by {err:.2e}")
    clamp = 1.0 - 2 * _FLOOR  # ON - OFF at the clamp, less the other plane's floor
    on_clip = float((d >= clamp)[x > 1e-6].mean())
    off_clip = float((d <= -clamp)[x < -1e-6].mean())

    def clip_shares(d: NDArray[np.float64]) -> dict[str, float]:
        """Share of the ON (x > 0) and OFF (x < 0) pixels of clean x that reach the clamp."""
        return {
            "on_clipped": float((d >= 1.0)[x > 1e-6].mean()),
            "off_clipped": float((d <= -1.0)[x < -1e-6].mean()),
        }

    # the gain at which normalisation gives the median clean digit the same RMS(d) as the
    # linear template: computed from x with the formula the run above has just confirmed
    matched = round_sig(gain * (sigma + digit), 2)
    scale = (sigma + rms(x))[:, None, None]
    result = {
        "rule": "norm_sigma = geometric mean of the median RMS(x) of the noise-only plane "
        "and of the clean digits, two significant figures",
        "test_indices": [first, last],
        "evaluation": f"use test indices below {first}: these are the calibration images",
        "input_noise_range": noise,
        "norm_sigma": sigma,
        "margin_over_noise_only": sigma / floor,
        "margin_under_clean": digit / sigma,
        "clean_clipping": {
            "linear_at_template_gain": {"gain": gain, **clip_shares(gain * x)},
            "normalized_at_template_gain": {
                "gain": gain,
                "on_clipped": on_clip,
                "off_clipped": off_clip,
            },
            "normalized_at_matched_gain": {
                "gain": matched,
                **clip_shares(matched * x / scale),
            },
        },
        "rms_x": measured,
    }
    # six decimals: the blank plane's probe floor (< 1e-9) reads as the 0 it stands for
    args.out.write_text(json.dumps(rounded(result), indent=4) + "\n")

    print(f"{'condition':<18} {'p5':>8} {'median':>8} {'p95':>8} "
          f"{'/clean':>7} {'/noise':>7} {'/sigma':>7} {'x/(s+x)':>7}")  # fmt: skip
    for name, s in measured.items():
        m = s["median"]
        print(
            f"{name:<18} {s['p5']:8.4f} {m:8.4f} {s['p95']:8.4f} "
            f"{m / digit:7.3f} {m / floor:7.2f} {m / sigma:7.2f} "
            f"{m / (sigma + m):7.3f}"
        )
    print(f"norm_sigma = {sigma}  (noise-only median {floor:.5f}, clean median {digit:.5f}); "
          f"normalised check max err {err:.1e}; at gain {gain}: ON clipped {on_clip:.1%}, "
          f"OFF clipped {off_clip:.1%}; matched gain {matched}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
