# coast
Coherent active state dynamics in self-organizing systems

## Dependencies

- CMake 3.15+
- SDL2
- Intel TBB
- HDF5

On macOS, GCC 15 is required for C++23 features (default clang lacks some):
```bash
brew install gcc@15 sdl2 tbb hdf5
```

## Build

```bash
# macOS (requires GCC 15)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(brew --prefix gcc)/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=$(brew --prefix gcc)/bin/g++-15 \
  -DCMAKE_CXX_FLAGS=-march=native

# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native

# Build
cmake --build build -j
```

## Usage

```bash
# From the build directory:
cd build
./coast ../configs/mnist.json
./coast ../configs/dsprites.json
```

## Configuration

COAST supports JSON configuration files for experiment reproducibility.

See `configs/mnist.json` for a complete example. Key sections:

## Datasets

Download datasets into `data/`:

### MNIST

```bash
mkdir -p data/mnist && cd data/mnist
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
gunzip *.gz
```

### dSprites

```bash
mkdir -p data/dsprites && cd data/dsprites
curl -LO https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5
```
