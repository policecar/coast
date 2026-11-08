"""MNIST and MNIST-C as COAST reads them: IDX ubyte files under a data root, SHA256-pinned.

IDX layout (the COAST loader's contract, `src/io/mnist_db.cpp`): a big-endian magic
`00 00 08 <ndim>` (0x08 = uint8), `ndim` big-endian int32 dimensions, then the row-major
uint8 payload. Images are `ndim = 3` (count, rows, cols), labels `ndim = 1`.

Every path in the pin table is relative to a `data_root` the caller passes. The MNIST-C
files are the test split of Mu & Gilmer's release (Zenodo record 3239543) converted to
IDX by `scripts/fetch_mnist_c.py`; their labels are byte-identical to clean MNIST's. The
same script writes each corruption's tuning images (`corruption_tuning_images`): the rows of
the release's train split at `protocol.tuning_split(TUNING_POOL)`, pinned in `TUNING_PINS`.
Those pins are kept apart from `PINS`, which every evaluation verifies, so that a data root
without the tuning files still serves an evaluation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

MNIST_C: tuple[str, ...] = (
    "brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "impulse_noise",
    "motion_blur", "rotate", "scale", "shear", "shot_noise", "spatter", "stripe",
    "translate", "zigzag",
)  # fmt: skip

TEST_IMAGES = "mnist/t10k-images-idx3-ubyte"
TEST_LABELS = "mnist/t10k-labels-idx1-ubyte"
TRAIN_IMAGES = "mnist/train-images-idx3-ubyte"
TRAIN_LABELS = "mnist/train-labels-idx1-ubyte"

_TEST_LABELS_SHA = "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2"
_MNIST_C_IMAGES_SHA = {
    "brightness": "43bd5017c4e418091d00fdaabee3e4b24903b785cb57940e22039046d9dd64d2",
    "canny_edges": "724e4b13946d3c8f8c9dd0c5344e3a0728076950d26db2be1e22034f2b75d17c",
    "dotted_line": "faa28a63a46189bbd9290f348051b41d4427a586a80febfc78646f2fb009a4e5",
    "fog": "2bca2db8ff19001f8a1b30d1e56a6c4c460ec85d7bf26045071e02752c240858",
    "glass_blur": "f4e615f4e6b2a991f78b9b7d1aa2da558466d40a1278e8187674787721976630",
    "impulse_noise": "a9d057e71120e1bbab00a85bb121fbef2ec8dce62590b0a8119ab882f47eb199",
    "motion_blur": "8861ae5f4f785d8ddff0f4102074fc8198bf4ac0664b6d9c418d6e04f79a63a9",
    "rotate": "166bd76fc16e3d1a075bbaf9868ce28ec99ed3c4d9ade6a83ee3f746fc3c2364",
    "scale": "cc7dbaddb76afd1818640aed67d3f0a48615ecee86761c5f55d72f3d7443aaa6",
    "shear": "72b60988d011c17faf56ad5685afc9ac1d3b451051755f9ed2494b8396929c9e",
    "shot_noise": "bd64a25cf943953c4e436fb19ea84fee378e0c471df5e2bcec82bb9c81fa8009",
    "spatter": "ec2beff53ad11579c9fef57ce4ec64173c64459c82e3417844015a830ea9ccab",
    "stripe": "6df8460090489daf384b341a841281cf73d85d2b0a0418e18ab38fca1cbf00d3",
    "translate": "6e586d96152b2a2fc05489454ef297cadcaf9a49b0e363991d0689011997921b",
    "zigzag": "6198b19cd3630b64c5d656be8b243b64a658647dd18b563f931ef874b4e9b0c3",
}


def corruption_files(name: str) -> tuple[str, str]:
    """(images, labels) of one MNIST-C corruption, relative to the data root."""
    if name not in MNIST_C:
        raise ValueError(f"unknown MNIST-C corruption '{name}'")
    return f"mnist_c/{name}/t10k-images-idx3-ubyte", f"mnist_c/{name}/t10k-labels-idx1-ubyte"


def corruption_tuning_images(name: str) -> str:
    """The tuning images of one MNIST-C corruption, relative to the data root: rows of the
    release's train split (`protocol.TuningInputs`). Their labels are clean MNIST's."""
    if name not in MNIST_C:
        raise ValueError(f"unknown MNIST-C corruption '{name}'")
    return f"mnist_c/{name}/tuning-images-idx3-ubyte"


# By fetch_mnist_c.py: each train_images.npy at rows protocol.tuning_split(TUNING_POOL) (numpy
# default_rng(TUNING_SEED).permutation), as write_idx IDX; re-derive if MNIST-C or numpy changes.
_MNIST_C_TUNING_SHA = {
    "brightness": "1d45b8d495e07d898e84ef1d49cc023a94c30f01985cff7fddbdc57ed2168c2a",
    "canny_edges": "56bf53ed2c3607125c70eb76e90f610cc84bd219fca01e36d9f920e1b988cd51",
    "dotted_line": "336a6fd61dbdaa27b78cd6e725134899c79fb520f1faeebaad4ef92fa840a4b8",
    "fog": "37ef79ddb89d0f6560798a4a3166f5e5e9bcffc33c16464e8249b506651fa834",
    "glass_blur": "a053ed897bb2825c0a052109f3b45b1bbe2bc29577eeb34bcbeb1f1fd84b2e73",
    "impulse_noise": "7e9c6515dd4054e03fc5919e16de3ee52cb874c679fb1393c4c19117b05ecc8d",
    "motion_blur": "f8cc0d451e924352addc6da9fb540f0d80b258e6072c0952615e165749b7607a",
    "rotate": "7a2bbc806c48ae8845f095a1a7a01e96bb0f793c72cd3a5aced109256541b86d",
    "scale": "3895c7a1f71957e1a42783aae1363cb7155be97dc6739fabc78acc12b67d467c",
    "shear": "0e9e819ca0309d3730e1b3ddc44cbb2648fa0b330ef42b204ec6ca9f9e72e6ed",
    "shot_noise": "7e74883169dcffa94e124dad4caa87475d48f57a06d25a460090dca6289a604f",
    "spatter": "c16f7b9d972c3fb40bfe26e24ab932fb0caaf2bed155750227db6770f9f7d779",
    "stripe": "610d0a95fddca9530f9f379969be249a07c86de8de6990d7193ceea104ce1ed2",
    "translate": "5255f91b5eb32732633f3ccf300a397f9ac9bce2be5c1fa61c9ca0f87e748905",
    "zigzag": "51dcd27d5cc08036c991c83a3ee2624fb803f9c77c2d975bd1010f08ee2f2bbd",
}

TUNING_PINS: Mapping[str, str] = MappingProxyType(
    {corruption_tuning_images(c): sha for c, sha in _MNIST_C_TUNING_SHA.items()}
)

PINS: Mapping[str, str] = MappingProxyType(
    {
        TRAIN_IMAGES: "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
        TRAIN_LABELS: "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
        TEST_IMAGES: "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
        TEST_LABELS: _TEST_LABELS_SHA,
    }
    | {corruption_files(c)[0]: sha for c, sha in _MNIST_C_IMAGES_SHA.items()}
    | {corruption_files(c)[1]: _TEST_LABELS_SHA for c in MNIST_C}
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_pins(data_root: Path, pins: Mapping[str, str]) -> None:
    """Raise ValueError naming the first file under `data_root` that is missing or whose
    SHA256 differs from its pin."""
    for rel, expected in pins.items():
        path = data_root / rel
        if not path.is_file():
            raise ValueError(f"{path}: missing (pinned {expected})")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} does not match the pin {expected}")


def _read_idx(path: Path, ndim: int) -> NDArray[np.uint8]:
    raw = path.read_bytes()
    if raw[:4] != bytes((0, 0, 0x08, ndim)):
        raise ValueError(f"{path}: magic {raw[:4].hex()} is not uint8 IDX with {ndim} dimensions")
    if len(raw) < 4 + 4 * ndim:
        raise ValueError(f"{path}: {len(raw)} bytes is shorter than an IDX header")
    dims = tuple(int(d) for d in np.frombuffer(raw, dtype=">i4", count=ndim, offset=4))
    payload = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if payload.size != int(np.prod(dims)):
        raise ValueError(f"{path}: payload of {payload.size} bytes for dimensions {dims}")
    return payload.reshape(dims)


def read_idx_images(path: Path) -> NDArray[np.uint8]:
    """(count, rows, cols) uint8 images; raises on a wrong magic, dimension count or size."""
    return _read_idx(path, 3)


def read_idx_labels(path: Path) -> NDArray[np.uint8]:
    """(count,) uint8 labels; raises on a wrong magic, dimension count or size."""
    return _read_idx(path, 1)


def write_idx(path: Path, a: NDArray[np.uint8]) -> None:
    """Write a uint8 array of 1 (labels) or 3 (images) dimensions as IDX."""
    if a.dtype != np.uint8 or a.ndim not in (1, 3) or a.size == 0:
        raise ValueError(f"write_idx: need a non-empty uint8 1-D or 3-D array, got {a.dtype}")
    header = bytes((0, 0, 0x08, a.ndim)) + np.array(a.shape, dtype=">i4").tobytes()
    path.write_bytes(header + np.ascontiguousarray(a).tobytes())
