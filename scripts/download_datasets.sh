#!/bin/bash
#
# Download script for CIMC-COAST datasets
# This script downloads all required datasets for the project.
# It checks for existing data and only downloads what's missing.
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found. Please install it."
        exit 1
    fi
}

check_python_module() {
    if python3 -c "import $1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check required commands
check_command curl
check_command gunzip
check_command unzip
check_command tar
check_command python3

# Check Python modules (needed for dSprites conversion)
HAS_NUMPY=false
HAS_H5PY=false
if check_python_module numpy; then
    HAS_NUMPY=true
fi
if check_python_module h5py; then
    HAS_H5PY=true
fi

# Create data directory structure
create_directories() {
    log_info "Creating data directory structure..."

    mkdir -p "$DATA_DIR/mnist"
    mkdir -p "$DATA_DIR/dsprites"
    mkdir -p "$DATA_DIR/emnist"
    mkdir -p "$DATA_DIR/kmnist"
    mkdir -p "$DATA_DIR/cifar10"
    mkdir -p "$DATA_DIR/cifar100"
    mkdir -p "$DATA_DIR/3dchairs"

    log_info "Directory structure created."
}

# Download MNIST
download_mnist() {
    local dir="$DATA_DIR/mnist"
    local images_file="$dir/train-images-idx3-ubyte"
    local labels_file="$dir/train-labels-idx1-ubyte"

    if [[ -f "$images_file" && -f "$labels_file" ]]; then
        log_info "MNIST: Already downloaded."
        return 0
    fi

    log_info "MNIST: Downloading..."
    cd "$dir"

    # Use ossci-datasets mirror (more reliable than original lecun.com)
    if [[ ! -f "$images_file" ]]; then
        curl -L --progress-bar -o train-images-idx3-ubyte.gz \
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz" || \
        curl -L --progress-bar -o train-images-idx3-ubyte.gz \
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        gunzip -f train-images-idx3-ubyte.gz
    fi

    if [[ ! -f "$labels_file" ]]; then
        curl -L --progress-bar -o train-labels-idx1-ubyte.gz \
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz" || \
        curl -L --progress-bar -o train-labels-idx1-ubyte.gz \
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        gunzip -f train-labels-idx1-ubyte.gz
    fi

    log_info "MNIST: Download complete."
}

# Download dSprites
download_dsprites() {
    local dir="$DATA_DIR/dsprites"
    local hdf5_file="$dir/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

    if [[ -f "$hdf5_file" ]]; then
        log_info "dSprites: Already downloaded."
        return 0
    fi

    # Check for required Python modules
    if [[ "$HAS_NUMPY" != "true" || "$HAS_H5PY" != "true" ]]; then
        log_warn "dSprites: Skipping - requires Python modules numpy and h5py."
        log_warn "         Install with: pip install numpy h5py"
        return 0
    fi

    log_info "dSprites: Downloading..."
    cd "$dir"

    # Download the npz file and convert to hdf5
    local npz_file="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    curl -L --progress-bar -o "$npz_file" \
        "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    # Convert npz to hdf5 using Python
    log_info "dSprites: Converting NPZ to HDF5..."
    python3 << 'PYTHON_SCRIPT'
import numpy as np
import h5py

npz_path = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
hdf5_path = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

data = np.load(npz_path, allow_pickle=True, encoding='latin1')

with h5py.File(hdf5_path, 'w') as f:
    for key in data.files:
        f.create_dataset(key, data=data[key])
PYTHON_SCRIPT

    # Clean up npz file
    rm -f "$npz_file"

    log_info "dSprites: Download complete."
}

# Download EMNIST
download_emnist() {
    local dir="$DATA_DIR/emnist"
    local images_file="$dir/emnist-byclass-train-images-idx3-ubyte"
    local labels_file="$dir/emnist-byclass-train-labels-idx1-ubyte"

    if [[ -f "$images_file" && -f "$labels_file" ]]; then
        log_info "EMNIST: Already downloaded."
        return 0
    fi

    log_info "EMNIST: Downloading (this may take a while, ~560MB)..."
    cd "$dir"

    curl -L --progress-bar -o emnist.zip \
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

    log_info "EMNIST: Extracting..."
    unzip -q emnist.zip

    gunzip -f gzip/emnist-byclass-train-images-idx3-ubyte.gz \
              gzip/emnist-byclass-train-labels-idx1-ubyte.gz

    mv gzip/emnist-byclass-train-images-idx3-ubyte .
    mv gzip/emnist-byclass-train-labels-idx1-ubyte .

    # Clean up
    rm -rf gzip emnist.zip

    log_info "EMNIST: Download complete."
}

# Download KMNIST
download_kmnist() {
    local dir="$DATA_DIR/kmnist"
    local images_file="$dir/kmnist-train-images-idx3-ubyte"
    local labels_file="$dir/kmnist-train-labels-idx1-ubyte"

    if [[ -f "$images_file" && -f "$labels_file" ]]; then
        log_info "KMNIST: Already downloaded."
        return 0
    fi

    log_info "KMNIST: Downloading..."
    cd "$dir"

    # Using CODH (Center for Open Data in the Humanities) - the original source
    curl -L -o train-images.gz \
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz"
    curl -L -o train-labels.gz \
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz"

    gunzip -f train-images.gz train-labels.gz

    mv train-images kmnist-train-images-idx3-ubyte
    mv train-labels kmnist-train-labels-idx1-ubyte

    log_info "KMNIST: Download complete."
}

# Download CIFAR-10
download_cifar10() {
    local dir="$DATA_DIR/cifar10"
    local batch_file="$dir/data_batch_1.bin"

    if [[ -f "$batch_file" ]]; then
        log_info "CIFAR-10: Already downloaded."
        return 0
    fi

    log_info "CIFAR-10: Downloading..."
    cd "$dir"

    curl -L --progress-bar -o cifar-10-binary.tar.gz \
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

    log_info "CIFAR-10: Extracting..."
    tar -xzf cifar-10-binary.tar.gz
    mv cifar-10-batches-bin/* .
    rmdir cifar-10-batches-bin
    rm cifar-10-binary.tar.gz

    log_info "CIFAR-10: Download complete."
}

# Download CIFAR-100
download_cifar100() {
    local dir="$DATA_DIR/cifar100"
    local train_file="$dir/train.bin"

    if [[ -f "$train_file" ]]; then
        log_info "CIFAR-100: Already downloaded."
        return 0
    fi

    log_info "CIFAR-100: Downloading..."
    cd "$dir"

    curl -L --progress-bar -o cifar-100-binary.tar.gz \
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"

    log_info "CIFAR-100: Extracting..."
    tar -xzf cifar-100-binary.tar.gz
    mv cifar-100-binary/* .
    rmdir cifar-100-binary
    rm cifar-100-binary.tar.gz

    log_info "CIFAR-100: Download complete."
}

# Download 3D Chairs
download_3dchairs() {
    local dir="$DATA_DIR/3dchairs"
    local chairs_dir="$dir/rendered_chairs"

    if [[ -d "$chairs_dir" ]]; then
        log_info "3D Chairs: Already downloaded."
        return 0
    fi

    log_info "3D Chairs: Downloading (~200MB, extraction takes time)..."
    cd "$dir"

    curl -L --progress-bar -o rendered_chairs.tar \
        "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"

    log_info "3D Chairs: Extracting (this may take a while)..."
    tar -xf rendered_chairs.tar
    rm rendered_chairs.tar

    log_info "3D Chairs: Download complete."
}

# Main function
main() {
    echo "========================================"
    echo "  CIMC-COAST Dataset Download Script"
    echo "========================================"
    echo ""

    log_info "Data directory: $DATA_DIR"
    echo ""

    # Create directories
    create_directories
    echo ""

    # Download all datasets
    download_mnist
    download_dsprites
    download_emnist
    download_kmnist
    download_cifar10
    download_cifar100
    download_3dchairs

    echo ""
    echo "========================================"
    log_info "All datasets ready!"
    echo "========================================"
}

# Run main function
main "$@"
