#!/usr/bin/env python3
"""Write noise variants of each dataset in the loader's native format.

kinds:
  perm  real images, pixel positions independently permuted per image (colour
        triplets kept together). Same intensity histogram => same input NSE as
        the real data, but no spatial structure.
  unif  iid uniform pixel values (dsprites: Bernoulli at the real pixel density).

Usage: gen_noise.py <data_root> <out_root> [--count N] [--seed S]
Writes <out_root>/<dataset>_<kind>/... and prints nothing on success.
"""
import argparse
import os
import struct

import h5py
import numpy as np
from PIL import Image

IDX = {
    "mnist": "mnist/train-images-idx3-ubyte",
    "emnist": "emnist/emnist-byclass-train-images-idx3-ubyte",
    "kmnist": "kmnist/kmnist-train-images-idx3-ubyte",
}
DSPRITES = "dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"


def read_idx3(path, count, rng):
    with open(path, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"not idx3: {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
    return data[rng.choice(n, count, replace=False)], r, c


def write_idx(dir_, imgs):
    os.makedirs(dir_, exist_ok=True)
    n, r, c = imgs.shape
    with open(os.path.join(dir_, "images-idx3-ubyte"), "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, r, c))
        f.write(imgs.astype(np.uint8).tobytes())
    with open(os.path.join(dir_, "labels-idx1-ubyte"), "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(np.zeros(n, np.uint8).tobytes())


def permute_positions(imgs, rng):
    """imgs: (n, P, C) -> each image's P positions permuted independently."""
    out = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        out[i] = imgs[i, rng.permutation(imgs.shape[1])]
    return out


def gen_idx(name, data_root, out_root, count, rng):
    imgs, r, c = read_idx3(os.path.join(data_root, IDX[name]), count, rng)
    flat = imgs.reshape(count, r * c, 1)
    write_idx(os.path.join(out_root, f"{name}_perm"),
              permute_positions(flat, rng).reshape(count, r, c))
    write_idx(os.path.join(out_root, f"{name}_unif"),
              rng.integers(0, 256, (count, r, c), dtype=np.uint8))


def gen_cifar10(data_root, out_root, count, rng):
    # loader with use_test_set reads exactly 10000 samples from test_batch.bin
    assert count <= 10000
    rec = 1 + 3072
    raw = np.fromfile(os.path.join(data_root, "cifar10/test_batch.bin"), np.uint8)
    raw = raw.reshape(10000, rec)
    planar = raw[:, 1:].reshape(10000, 3, 1024).transpose(0, 2, 1)  # (n, P, C)
    sel = rng.choice(10000, 10000, replace=False)
    variants = {
        "perm": permute_positions(planar[sel], rng),
        "unif": rng.integers(0, 256, planar.shape, dtype=np.uint8),
    }
    for kind, px in variants.items():
        d = os.path.join(out_root, f"cifar10_{kind}")
        os.makedirs(d, exist_ok=True)
        out = np.zeros((10000, rec), np.uint8)
        out[:, 1:] = px.transpose(0, 2, 1).reshape(10000, 3072)
        out.tofile(os.path.join(d, "test_batch.bin"))


def gen_dsprites(data_root, out_root, count, rng):
    with h5py.File(os.path.join(data_root, DSPRITES), "r") as f:
        ds = f["imgs"]
        idx = np.sort(rng.choice(ds.shape[0], count, replace=False))
        imgs = ds[idx]
    n, h, w = imgs.shape
    density = imgs.mean()
    variants = {
        "perm": permute_positions(imgs.reshape(n, h * w, 1), rng).reshape(n, h, w),
        "unif": (rng.random((n, h, w)) < density).astype(imgs.dtype),
    }
    for kind, px in variants.items():
        d = os.path.join(out_root, f"dsprites_{kind}")
        os.makedirs(d, exist_ok=True)
        with h5py.File(os.path.join(d, "imgs.hdf5"), "w") as f:
            f.create_dataset("imgs", data=px)


def gen_chairs(data_root, out_root, count, rng, size=64):
    files = []
    for root, _, names in os.walk(os.path.join(data_root, "3dchairs")):
        files += [os.path.join(root, n) for n in names if n.endswith(".png")]
    files.sort()
    pick = [files[i] for i in rng.choice(len(files), count, replace=False)]
    imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size),
                                                                     Image.BILINEAR))
                     for p in pick])
    variants = {
        "perm": permute_positions(imgs.reshape(count, size * size, 3), rng)
        .reshape(count, size, size, 3),
        "unif": rng.integers(0, 256, imgs.shape, dtype=np.uint8),
    }
    for kind, px in variants.items():
        d = os.path.join(out_root, f"chairs3d_{kind}")
        os.makedirs(d, exist_ok=True)
        for i in range(count):
            Image.fromarray(px[i]).save(os.path.join(d, f"img_{i:05d}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("out_root")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for name in IDX:
        gen_idx(name, a.data_root, a.out_root, a.count, rng)
    gen_cifar10(a.data_root, a.out_root, a.count, rng)
    gen_dsprites(a.data_root, a.out_root, a.count, rng)
    gen_chairs(a.data_root, a.out_root, a.count, rng)


if __name__ == "__main__":
    main()
