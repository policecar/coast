"""IDX readers and SHA pins on small synthetic files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from coast_eval import datasets as ds


def images(n: int = 3, rows: int = 4, cols: int = 5) -> NDArray[np.uint8]:
    return (np.arange(n * rows * cols) % 251).astype(np.uint8).reshape(n, rows, cols)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 4, 5), (300, 2, 3)])
def test_images_round_trip(tmp_path: Path, shape: tuple[int, int, int]) -> None:
    a = images(*shape)
    ds.write_idx(tmp_path / "i", a)
    assert np.array_equal(ds.read_idx_images(tmp_path / "i"), a)


def test_labels_round_trip(tmp_path: Path) -> None:
    a = np.array([7, 0, 255, 3], dtype=np.uint8)
    ds.write_idx(tmp_path / "l", a)
    assert np.array_equal(ds.read_idx_labels(tmp_path / "l"), a)


def test_header_is_big_endian(tmp_path: Path) -> None:
    ds.write_idx(tmp_path / "i", images(n=258, rows=2, cols=1))
    assert (tmp_path / "i").read_bytes()[:16] == bytes.fromhex(
        "00000803 00000102 00000002 00000001"
    )


MALFORMED: dict[str, tuple[Callable[[bytes], bytes], str]] = {
    "int8 payload": (lambda b: b[:2] + b"\x09" + b[3:], "magic"),
    "nonzero top byte": (lambda b: b"\x01" + b[1:], "magic"),
    "truncated": (lambda b: b[:-1], "payload"),
    "trailing bytes": (lambda b: b + b"\x00", "payload"),
    "header cut short": (lambda b: b[:10], "header"),
}


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_images_raise(tmp_path: Path, case: str) -> None:
    edit, match = MALFORMED[case]
    path = tmp_path / "i"
    ds.write_idx(path, images())
    path.write_bytes(edit(path.read_bytes()))
    with pytest.raises(ValueError, match=match):
        ds.read_idx_images(path)


def test_dimension_count_is_checked(tmp_path: Path) -> None:
    ds.write_idx(tmp_path / "i", images())
    ds.write_idx(tmp_path / "l", np.arange(7, dtype=np.uint8))
    with pytest.raises(ValueError, match="3 dimensions"):
        ds.read_idx_images(tmp_path / "l")
    with pytest.raises(ValueError, match="1 dimensions"):
        ds.read_idx_labels(tmp_path / "i")


@pytest.mark.parametrize(
    "a", [np.zeros((2, 3), np.uint8), np.zeros((2, 2, 2), np.int16), np.zeros((0,), np.uint8)]
)
def test_write_refuses_what_coast_cannot_read(tmp_path: Path, a: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError, match="write_idx"):
        ds.write_idx(tmp_path / "x", a)  # type: ignore[arg-type]


def test_verify_pins_names_the_first_bad_file(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one").write_bytes(b"one")
    (tmp_path / "two").write_bytes(b"two")
    pins = {
        "a/one": ds.sha256_file(tmp_path / "a" / "one"),
        "two": ds.sha256_file(tmp_path / "two"),
    }
    ds.verify_pins(tmp_path, pins)
    (tmp_path / "two").write_bytes(b"tw0")
    with pytest.raises(ValueError, match=r"two: sha256 .* does not match the pin"):
        ds.verify_pins(tmp_path, pins)
    (tmp_path / "a" / "one").unlink()
    with pytest.raises(ValueError, match=r"one: missing"):
        ds.verify_pins(tmp_path, pins)


def test_pin_table_covers_mnist_and_every_corruption() -> None:
    assert len(ds.MNIST_C) == 15
    assert len(ds.PINS) == 4 + 2 * 15
    for c in ds.MNIST_C:
        images_rel, labels_rel = ds.corruption_files(c)
        assert ds.PINS[labels_rel] == ds.PINS[ds.TEST_LABELS]
        assert ds.PINS[images_rel] != ds.PINS[ds.TEST_IMAGES]
    with pytest.raises(ValueError, match="unknown MNIST-C"):
        ds.corruption_files("identity")
