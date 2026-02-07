+++
title = "io"
weight = 40
+++

# io — Dataset Loaders

The `io` module implements dataset entities — source entities with no inputs that present training data to the simulation one sample at a time.

## Directory Layout

```
include/io/
├── dataset_cfg.h             # Common dataset type enum
├── mnist_cfg.h / mnist_io.h  # MNIST handwritten digits (28x28x1)
├── emnist_cfg.h / emnist_io.h # Extended MNIST
├── kmnist_cfg.h / kmnist_io.h # Kuzushiji MNIST
├── cifar10_cfg.h / cifar10_io.h # CIFAR-10 (32x32x3 RGB)
├── cifar100_cfg.h / cifar100_io.h # CIFAR-100
├── dsprites_cfg.h / dsprites_io.h # dSprites (64x64x1, HDF5)
├── chairs3d_cfg.h / chairs3d_io.h # 3D Chairs
├── color_mnist_cfg.h / color_mnist_io.h # Colorized MNIST
└── (each _cfg.h has a matching _param_t struct)

src/io/
├── mnist_db.cpp              # IDX file format reader
├── mnist_io.cpp              # MNIST io_entity implementation
├── cifar10_io.cpp            # CIFAR-10 binary format reader
├── dsprites_io.cpp           # HDF5-based loader
├── ... (one .cpp per dataset)
```

## Common Pattern

All dataset entities follow the same pattern. They are **source entities** with no inputs:

### Constructor
1. Load dataset from disk into memory (normalized to `[0,1]` floats)
2. Initialize RNG with configured seed
3. Initialize noise distribution from `noise_range`

### process()
1. Get write buffer via `output_mem()`
2. Copy current sample into buffer
3. Add noise: `clamp(val + noise, 0, 1)` per element
4. Advance counter; wrap to next sample at `change_interval`

### Configuration Parameters

Each `*_param_t` struct contains:

| Parameter | Purpose |
|-----------|---------|
| Data path(s) | Where to find the dataset files |
| `change_interval` | Steps between sample changes (0 = hold) |
| `noise_range` | Uniform noise bounds (min, max) |
| `random_seed` | RNG seed for reproducibility |

Dataset-specific parameters include `use_test_set` (CIFAR), `randomize_readout` (dSprites), `image_width/height` (Chairs3D).

### Common API

All datasets provide (informally, not via virtual base):
- `get_image_info()` → (width, height, channels)
- `sample_count()`, `current_epoch()`, `current_index()`
- `get_sample(idx)` → `span<const float>`

### State Persistence

`save_state()` / `load_state()` persist `cur_epoch`, `cur_idx`, `ci_cnt` so the dataset resumes from where it left off.

## Dataset Specifics

| Dataset | Image Size | Format | Notes |
|---------|-----------|--------|-------|
| MNIST | 28x28x1 | IDX binary | Inherits from `mnist_db` for file reading |
| EMNIST | 28x28x1 | IDX binary | Same format as MNIST |
| KMNIST | 28x28x1 | IDX binary | Same format as MNIST |
| CIFAR-10 | 32x32x3 | Binary batches | Planar→interleaved RGB conversion |
| CIFAR-100 | 32x32x3 | Binary | Optional coarse/fine labels |
| dSprites | 64x64x1 | HDF5 | Binary images (0/1 → 0.0/1.0) |
| Chairs3D | configurable | Images | Loaded via stb_image |
| Color MNIST | 28x28x3 | IDX binary | Colorized variant |
