#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "coast-eval",
#   "numpy",
# ]
#
# [tool.uv.sources]
# coast-eval = { path = "../eval", editable = true }
# ///
"""Provision MNIST-C under a data root as the IDX files COAST reads, checked against the pins.

Usage:
    scripts/fetch_mnist_c.py --data-root data                # download if needed
    scripts/fetch_mnist_c.py --data-root data --zip mnist_c.zip

1. If every pinned MNIST-C file under <data-root>/mnist_c/ already matches its pin
   (coast_eval.datasets.PINS and TUNING_PINS), exit without touching anything.
2. Otherwise take --zip, or download Mu & Gilmer's release (Zenodo record 3239543) to
   <data-root>/mnist_c/mnist_c.zip; either way its MD5 must match Zenodo's.
3. Convert each corruption's test split (test_images.npy, (10000, 28, 28, 1) uint8;
   test_labels.npy, int64 0..9) to IDX. A file that matches its pin is never rewritten;
   a present file that does not match is refused (remove it by hand); each new file is
   written beside its target, checked against its pin, then renamed into place.
4. The tuning images of a sweep (coast_eval.protocol.TuningInputs): per corruption, the rows
   of the release's train split (train_images.npy, (60000, 28, 28, 1)) at
   protocol.tuning_split(TUNING_POOL), in that order, as mnist_c/<c>/tuning-images-idx3-ubyte.
   The release's train labels must equal clean MNIST's, so the rows are the same images.
   Each is handled as in 3.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from coast_eval import datasets as ds
from coast_eval import protocol

URL = "https://zenodo.org/records/3239543/files/mnist_c.zip"
ZIP_MD5 = "4b34b33045869ee6d424616cd3a65da3"  # Zenodo's checksum for mnist_c.zip


def missing_files(data_root: Path, pins: dict[str, str]) -> list[str]:
    """Pinned files not yet present; raises on a present file that does not match."""
    missing: list[str] = []
    for rel, sha in pins.items():
        path = data_root / rel
        if not path.exists():
            missing.append(rel)
        elif ds.sha256_file(path) != sha:
            raise SystemExit(f"{path}: present but not the pinned {sha}; refusing to overwrite")
    return missing


def archive(data_root: Path, given: Path | None) -> Path:
    path = src = given or data_root / "mnist_c" / "mnist_c.zip"
    if given is None and not path.exists():
        src = path.with_name(path.name + ".part")
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, src)
    with src.open("rb") as f:
        md5 = hashlib.file_digest(f, "md5").hexdigest()
    if md5 != ZIP_MD5:
        raise SystemExit(f"{src}: md5 {md5} is not the release's {ZIP_MD5}")
    src.rename(path)  # a download takes the final name only once it is the release
    return path


def converted(z: zipfile.ZipFile, rel: str) -> NDArray[np.uint8]:
    """The IDX content of `rel` (mnist_c/<name>/t10k-...) from the archive's test split."""
    name, kind = rel.split("/")[1], "images" if "images" in rel else "labels"
    a = np.load(io.BytesIO(z.read(f"mnist_c/{name}/test_{kind}.npy")))
    if kind == "images":
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[1:] != (28, 28, 1):
            raise SystemExit(f"{name}: test images are {a.dtype} {a.shape}")
        return a[..., 0]
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 9:
        raise SystemExit(f"{name}: test labels are not integers in 0..9")
    return a.astype(np.uint8)


def converted_tuning(
    z: zipfile.ZipFile, name: str, clean_labels: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """The tuning images of corruption `name`: train-split rows at the tuning split's pool."""
    a = np.load(io.BytesIO(z.read(f"mnist_c/{name}/train_images.npy")))
    if a.dtype != np.uint8 or a.shape != (protocol.TRAIN_COUNT, 28, 28, 1):
        raise SystemExit(f"{name}: train images are {a.dtype} {a.shape}")
    labels = np.load(io.BytesIO(z.read(f"mnist_c/{name}/train_labels.npy")))
    if labels.shape != clean_labels.shape or not np.array_equal(labels, clean_labels):
        raise SystemExit(f"{name}: train labels differ from clean MNIST's: not its image order")
    return a[np.asarray(protocol.tuning_split(protocol.TUNING_POOL)), ..., 0]


def clean_train_labels(data_root: Path) -> NDArray[np.uint8]:
    ds.verify_pins(data_root, {ds.TRAIN_LABELS: ds.PINS[ds.TRAIN_LABELS]})
    return ds.read_idx_labels(data_root / ds.TRAIN_LABELS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--zip", type=Path, help="a local copy of mnist_c.zip (no download)")
    args = ap.parse_args()
    tuning = {ds.corruption_tuning_images(c): c for c in ds.MNIST_C}
    pins = {rel: sha for rel, sha in ds.PINS.items() if rel.startswith("mnist_c/")}
    pins |= dict(ds.TUNING_PINS)
    todo = missing_files(args.data_root, pins)
    if not todo:
        print(f"all {len(pins)} MNIST-C files under {args.data_root} match their pins")
        return 0
    labels = clean_train_labels(args.data_root) if any(r in tuning for r in todo) else None
    with zipfile.ZipFile(archive(args.data_root, args.zip)) as z:
        for rel in todo:
            if rel in tuning:
                assert labels is not None
                content = converted_tuning(z, tuning[rel], labels)
            else:
                content = converted(z, rel)
            path = args.data_root / rel
            part = path.with_name(path.name + ".part")
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.write_idx(part, content)
            if ds.sha256_file(part) != pins[rel]:
                part.unlink()
                raise SystemExit(f"{rel}: converted file does not match its pin")
            part.rename(path)
            print(f"wrote {path}")
    ds.verify_pins(args.data_root, pins)
    print(f"all {len(pins)} MNIST-C files under {args.data_root} match their pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
