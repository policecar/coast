#!/bin/bash
#
# Download script for CIMC-COAST datasets
# This script downloads all required datasets for the project.
# It checks for existing data and only downloads what's missing.
#

# Note: Not using set -e as we handle errors in download functions

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ANSI cursor control
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'
CLEAR_LINE='\033[K'
HIDE_CURSOR='\033[?25l'
SHOW_CURSOR='\033[?25h'

# Global state for parallel downloads
declare -A DOWNLOAD_PIDS      # PID for each dataset download
declare -A DOWNLOAD_STATUS    # Status: downloading, extracting, done, failed, skipped
declare -A DOWNLOAD_PROGRESS  # Progress percentage
declare -A DOWNLOAD_SPEED     # Download speed in bytes/sec
ABORT_DOWNLOADS=false
DOWNLOAD_START_LINE=0

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

# Move cursor to specific line (relative to download start)
move_to_line() {
    local line=$1
    printf '\033[%d;0H' "$((DOWNLOAD_START_LINE + line))"
}

# Draw a progress bar
draw_progress_bar() {
    local percent=$1
    local width=30
    local filled=$((percent * width / 100))
    local empty=$((width - filled))
    printf '['
    printf '%*s' "$filled" '' | tr ' ' '#'
    printf '%*s' "$empty" '' | tr ' ' '-'
    printf '] %3d%%' "$percent"
}

# Update status display for a dataset
update_dataset_status() {
    local dataset=$1
    local line=$2
    local status="${DOWNLOAD_STATUS[$dataset]}"
    local progress="${DOWNLOAD_PROGRESS[$dataset]:-0}"
    local speed="${DOWNLOAD_SPEED[$dataset]:-0}"

    # Get display name
    local name
    IFS='|' read -r name _ _ <<< "${DATASET_INFO[$dataset]}"

    move_to_line "$line"
    printf "${CLEAR_LINE}"

    case "$status" in
        waiting)
            printf "  %-13s ${YELLOW}waiting...${NC}" "$name"
            ;;
        downloading)
            printf "  %-13s ${CYAN}" "$name"
            draw_progress_bar "$progress"
            if [[ $speed -gt 0 ]]; then
                printf " %s" "$(format_speed "$speed")"
            fi
            printf "${NC}"
            ;;
        extracting)
            printf "  %-13s ${YELLOW}extracting...${NC}" "$name"
            ;;
        done)
            printf "  %-13s ${GREEN}done${NC}" "$name"
            ;;
        failed)
            printf "  %-13s ${RED}failed${NC}" "$name"
            ;;
        skipped)
            printf "  %-13s ${YELLOW}skipped (missing dependencies)${NC}" "$name"
            ;;
    esac
}

# Cleanup function for abort
cleanup_downloads() {
    ABORT_DOWNLOADS=true
    # Kill all download processes
    for pid in "${DOWNLOAD_PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    printf "${SHOW_CURSOR}"
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
    mkdir -p "$DATA_DIR/mnist"
    mkdir -p "$DATA_DIR/dsprites"
    mkdir -p "$DATA_DIR/emnist"
    mkdir -p "$DATA_DIR/kmnist"
    mkdir -p "$DATA_DIR/cifar10"
    mkdir -p "$DATA_DIR/cifar100"
    mkdir -p "$DATA_DIR/3dchairs"
}

# Status file location
STATUS_DIR=""

# Write status for a dataset
write_status() {
    local dataset=$1
    local status=$2
    local progress=${3:-0}
    local speed=${4:-0}
    echo "${status}:${progress}:${speed}" > "${STATUS_DIR}/${dataset}.status"
}

# Read status for a dataset
read_status() {
    local dataset=$1
    local status_file="${STATUS_DIR}/${dataset}.status"
    if [[ -f "$status_file" ]]; then
        cat "$status_file"
    else
        echo "waiting:0:0"
    fi
}

# Format bytes per second to human readable
format_speed() {
    local bytes_per_sec=$1
    if [[ $bytes_per_sec -ge 1048576 ]]; then
        awk "BEGIN {printf \"%.1f MB/s\", $bytes_per_sec / 1048576}"
    elif [[ $bytes_per_sec -ge 1024 ]]; then
        awk "BEGIN {printf \"%.0f KB/s\", $bytes_per_sec / 1024}"
    else
        printf "%d B/s" "$bytes_per_sec"
    fi
}

# Download a file with progress tracking
# Usage: download_file dataset url output_file expected_size
download_file() {
    local dataset=$1
    local url=$2
    local output=$3
    local expected_size=$4

    # Start curl in background, writing to file
    curl -L -s -o "$output" "$url" &
    local curl_pid=$!

    # Variables for speed calculation
    local prev_size=0
    local prev_time
    prev_time=$(date +%s%N)

    # Monitor progress
    while kill -0 "$curl_pid" 2>/dev/null; do
        if [[ "$ABORT_DOWNLOADS" == "true" ]]; then
            kill "$curl_pid" 2>/dev/null || true
            return 1
        fi

        if [[ -f "$output" && "$expected_size" -gt 0 ]]; then
            local current_size
            current_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null || echo 0)
            local current_time
            current_time=$(date +%s%N)

            # Calculate speed (bytes per second)
            local time_diff=$(( (current_time - prev_time) / 1000000 ))  # milliseconds
            local size_diff=$((current_size - prev_size))
            local speed=0
            if [[ $time_diff -gt 0 ]]; then
                speed=$((size_diff * 1000 / time_diff))  # bytes per second
            fi

            local percent=$((current_size * 100 / expected_size))
            [[ $percent -gt 100 ]] && percent=100

            write_status "$dataset" "downloading" "$percent" "$speed"

            prev_size=$current_size
            prev_time=$current_time
        fi
        sleep 0.3
    done

    wait "$curl_pid"
    return $?
}

# Download MNIST (with progress tracking)
download_mnist_tracked() {
    local dir="$DATA_DIR/mnist"
    local images_file="$dir/train-images-idx3-ubyte"
    local labels_file="$dir/train-labels-idx1-ubyte"

    if [[ -f "$images_file" && -f "$labels_file" ]]; then
        write_status "mnist" "done" 100
        return 0
    fi

    cd "$dir"
    write_status "mnist" "downloading" 0

    # Download files (~9.9MB images + ~29KB labels = ~10MB total)
    download_file "mnist" \
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz" \
        "train-images-idx3-ubyte.gz" 9912422 || {
        write_status "mnist" "failed" 0
        return 1
    }

    curl -L -s -o train-labels-idx1-ubyte.gz \
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"

    write_status "mnist" "extracting" 100
    gunzip -f train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 2>/dev/null || true

    write_status "mnist" "done" 100
}

# Download dSprites (with progress tracking)
download_dsprites_tracked() {
    local dir="$DATA_DIR/dsprites"
    local hdf5_file="$dir/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

    if [[ -f "$hdf5_file" ]]; then
        write_status "dsprites" "done" 100
        return 0
    fi

    # Check for required Python modules
    if [[ "$HAS_NUMPY" != "true" || "$HAS_H5PY" != "true" ]]; then
        write_status "dsprites" "skipped" 0
        return 0
    fi

    cd "$dir"
    write_status "dsprites" "downloading" 0

    local npz_file="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    # ~18MB download (expands to ~2.9GB)
    download_file "dsprites" \
        "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz" \
        "$npz_file" 19318135 || {
        write_status "dsprites" "failed" 0
        return 1
    }

    write_status "dsprites" "extracting" 100
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

    rm -f "$npz_file"
    write_status "dsprites" "done" 100
}

# Download EMNIST (with progress tracking)
download_emnist_tracked() {
    local dir="$DATA_DIR/emnist"
    local images_file="$dir/emnist-byclass-train-images-idx3-ubyte"
    local labels_file="$dir/emnist-byclass-train-labels-idx1-ubyte"

    if [[ -f "$images_file" && -f "$labels_file" ]]; then
        write_status "emnist" "done" 100
        return 0
    fi

    cd "$dir"
    write_status "emnist" "downloading" 0

    # ~535MB file
    download_file "emnist" \
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip" \
        "emnist.zip" 561753746 || {
        write_status "emnist" "failed" 0
        return 1
    }

    write_status "emnist" "extracting" 100
    unzip -q emnist.zip

    gunzip -f gzip/emnist-byclass-train-images-idx3-ubyte.gz \
              gzip/emnist-byclass-train-labels-idx1-ubyte.gz

    mv gzip/emnist-byclass-train-images-idx3-ubyte .
    mv gzip/emnist-byclass-train-labels-idx1-ubyte .

    rm -rf gzip emnist.zip
    write_status "emnist" "done" 100
}

# Download KMNIST (with progress tracking)
download_kmnist_tracked() {
    local dir="$DATA_DIR/kmnist"
    local images_file="$dir/kmnist-train-images-idx3-ubyte"
    local labels_file="$dir/kmnist-train-labels-idx1-ubyte"

    if [[ -f "$images_file" && -f "$labels_file" ]]; then
        write_status "kmnist" "done" 100
        return 0
    fi

    cd "$dir"
    write_status "kmnist" "downloading" 0

    # ~18MB (images file is ~18MB, labels is tiny)
    download_file "kmnist" \
        "https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz" \
        "train-images.gz" 18165135 || {
        write_status "kmnist" "failed" 0
        return 1
    }

    curl -L -s -o train-labels.gz \
        "https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz"

    write_status "kmnist" "extracting" 100
    gunzip -f train-images.gz train-labels.gz

    mv train-images kmnist-train-images-idx3-ubyte
    mv train-labels kmnist-train-labels-idx1-ubyte

    write_status "kmnist" "done" 100
}

# Download CIFAR-10 (with progress tracking)
download_cifar10_tracked() {
    local dir="$DATA_DIR/cifar10"
    local batch_file="$dir/data_batch_1.bin"

    if [[ -f "$batch_file" ]]; then
        write_status "cifar10" "done" 100
        return 0
    fi

    cd "$dir"
    write_status "cifar10" "downloading" 0

    # ~163MB
    download_file "cifar10" \
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" \
        "cifar-10-binary.tar.gz" 170052171 || {
        write_status "cifar10" "failed" 0
        return 1
    }

    write_status "cifar10" "extracting" 100
    tar -xzf cifar-10-binary.tar.gz
    mv cifar-10-batches-bin/* .
    rmdir cifar-10-batches-bin
    rm cifar-10-binary.tar.gz

    write_status "cifar10" "done" 100
}

# Download CIFAR-100 (with progress tracking)
download_cifar100_tracked() {
    local dir="$DATA_DIR/cifar100"
    local train_file="$dir/train.bin"

    if [[ -f "$train_file" ]]; then
        write_status "cifar100" "done" 100
        return 0
    fi

    cd "$dir"
    write_status "cifar100" "downloading" 0

    # ~161MB
    download_file "cifar100" \
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz" \
        "cifar-100-binary.tar.gz" 168513733 || {
        write_status "cifar100" "failed" 0
        return 1
    }

    write_status "cifar100" "extracting" 100
    tar -xzf cifar-100-binary.tar.gz
    mv cifar-100-binary/* .
    rmdir cifar-100-binary
    rm cifar-100-binary.tar.gz

    write_status "cifar100" "done" 100
}

# Download 3D Chairs (with progress tracking)
download_3dchairs_tracked() {
    local dir="$DATA_DIR/3dchairs"
    local chairs_dir="$dir/rendered_chairs"

    if [[ -d "$chairs_dir" ]]; then
        write_status "3dchairs" "done" 100
        return 0
    fi

    cd "$dir"
    write_status "3dchairs" "downloading" 0

    # ~2.2GB
    download_file "3dchairs" \
        "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar" \
        "rendered_chairs.tar" 2406451200 || {
        write_status "3dchairs" "failed" 0
        return 1
    }

    write_status "3dchairs" "extracting" 100
    tar -xf rendered_chairs.tar
    rm rendered_chairs.tar

    write_status "3dchairs" "done" 100
}

# Dataset information
# Format: name|download_size|installed_size|check_function
declare -A DATASET_INFO
DATASET_INFO["mnist"]="MNIST|11 MB|47 MB"
DATASET_INFO["dsprites"]="dSprites|18 MB|2.9 GB"
DATASET_INFO["emnist"]="EMNIST|535 MB|700 MB"
DATASET_INFO["kmnist"]="KMNIST|18 MB|47 MB"
DATASET_INFO["cifar10"]="CIFAR-10|163 MB|163 MB"
DATASET_INFO["cifar100"]="CIFAR-100|161 MB|161 MB"
DATASET_INFO["3dchairs"]="3D Chairs|2.2 GB|2.2 GB"

# Dataset order for display
DATASET_ORDER=("mnist" "dsprites" "emnist" "kmnist" "cifar10" "cifar100" "3dchairs")

# Selection state (1 = selected, 0 = not selected)
declare -A DATASET_SELECTED

# Check if a dataset is installed
is_dataset_installed() {
    local dataset="$1"
    case "$dataset" in
        mnist)
            [[ -f "$DATA_DIR/mnist/train-images-idx3-ubyte" && \
               -f "$DATA_DIR/mnist/train-labels-idx1-ubyte" ]]
            ;;
        dsprites)
            [[ -f "$DATA_DIR/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5" ]]
            ;;
        emnist)
            [[ -f "$DATA_DIR/emnist/emnist-byclass-train-images-idx3-ubyte" && \
               -f "$DATA_DIR/emnist/emnist-byclass-train-labels-idx1-ubyte" ]]
            ;;
        kmnist)
            [[ -f "$DATA_DIR/kmnist/kmnist-train-images-idx3-ubyte" && \
               -f "$DATA_DIR/kmnist/kmnist-train-labels-idx1-ubyte" ]]
            ;;
        cifar10)
            [[ -f "$DATA_DIR/cifar10/data_batch_1.bin" ]]
            ;;
        cifar100)
            [[ -f "$DATA_DIR/cifar100/train.bin" ]]
            ;;
        3dchairs)
            [[ -d "$DATA_DIR/3dchairs/rendered_chairs" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

# Get available disk space in human-readable format
get_available_space() {
    local target_dir="$DATA_DIR"
    if [[ ! -d "$target_dir" ]]; then
        target_dir="$PROJECT_ROOT"
    fi
    df -h "$target_dir" 2>/dev/null | awk 'NR==2 {print $4}'
}

# Get available disk space in bytes
get_available_space_bytes() {
    local target_dir="$DATA_DIR"
    if [[ ! -d "$target_dir" ]]; then
        target_dir="$PROJECT_ROOT"
    fi
    df --block-size=1 "$target_dir" 2>/dev/null | awk 'NR==2 {print $4}' || \
    df -k "$target_dir" 2>/dev/null | awk 'NR==2 {print $4 * 1024}'
}

# Clear screen and move cursor to top
clear_menu() {
    printf '\033[2J\033[H'
}

# Print the menu header
print_header() {
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}              CIMC-COAST Dataset Download Script${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "Data directory: ${YELLOW}$DATA_DIR${NC}"
    echo -e "Available disk space: ${YELLOW}$(get_available_space)${NC}"
    echo ""
}

# Print the dataset table
print_dataset_table() {
    echo "┌───────┬───────────────┬─────────────┬─────────────┬───────────┐"
    echo "│  Sel  │ Dataset       │ Download    │ Installed   │ Status    │"
    echo "├───────┼───────────────┼─────────────┼─────────────┼───────────┤"

    local idx=1
    for dataset in "${DATASET_ORDER[@]}"; do
        IFS='|' read -r name dl_size inst_size <<< "${DATASET_INFO[$dataset]}"

        # Check installation status
        local status_text status_color
        if is_dataset_installed "$dataset"; then
            status_text="Installed"
            status_color="${GREEN}"
        else
            status_text="Missing"
            status_color="${RED}"
        fi

        # Check selection
        local checkbox
        if [[ "${DATASET_SELECTED[$dataset]}" == "1" ]]; then
            checkbox="[x]"
        else
            checkbox="[ ]"
        fi

        # Build the row without colors first, then apply color to status
        # Using printf to pad status_text, then wrap with color codes
        local padded_status
        padded_status=$(printf "%-9s" "$status_text")

        printf "│ %s %d │ %-13s │ %11s │ %11s │ %b%s%b │\n" \
            "$checkbox" "$idx" "$name" "$dl_size" "$inst_size" \
            "$status_color" "$padded_status" "$NC"

        ((idx++))
    done

    echo "└───────┴───────────────┴─────────────┴─────────────┴───────────┘"
    echo ""
}

# Print menu instructions
print_instructions() {
    echo "Commands:"
    echo "  1-7     Toggle dataset selection"
    echo "  a       Select all missing datasets"
    echo "  n       Deselect all"
    echo "  d       Download selected datasets"
    echo "  q       Quit without downloading"
    echo ""
}

# Initialize selection state (pre-select missing datasets)
init_selection() {
    for dataset in "${DATASET_ORDER[@]}"; do
        if is_dataset_installed "$dataset"; then
            DATASET_SELECTED[$dataset]=0
        else
            DATASET_SELECTED[$dataset]=1
        fi
    done
}

# Toggle selection for a dataset
toggle_selection() {
    local idx=$1
    if [[ $idx -ge 1 && $idx -le ${#DATASET_ORDER[@]} ]]; then
        local dataset="${DATASET_ORDER[$((idx-1))]}"
        if [[ "${DATASET_SELECTED[$dataset]}" == "1" ]]; then
            DATASET_SELECTED[$dataset]=0
        else
            DATASET_SELECTED[$dataset]=1
        fi
    fi
}

# Select all missing datasets
select_all_missing() {
    for dataset in "${DATASET_ORDER[@]}"; do
        if ! is_dataset_installed "$dataset"; then
            DATASET_SELECTED[$dataset]=1
        fi
    done
}

# Deselect all datasets
deselect_all() {
    for dataset in "${DATASET_ORDER[@]}"; do
        DATASET_SELECTED[$dataset]=0
    done
}

# Count selected datasets
count_selected() {
    local count=0
    for dataset in "${DATASET_ORDER[@]}"; do
        if [[ "${DATASET_SELECTED[$dataset]}" == "1" ]]; then
            ((count++))
        fi
    done
    echo $count
}

# Download selected datasets with progress display
download_selected() {
    local selected_count
    selected_count=$(count_selected)

    if [[ $selected_count -eq 0 ]]; then
        log_warn "No datasets selected for download."
        read -p "Press Enter to continue..."
        return
    fi

    # Create temp directory for status files
    STATUS_DIR=$(mktemp -d)
    trap 'rm -rf "$STATUS_DIR"' EXIT

    # Reset abort flag
    ABORT_DOWNLOADS=false

    # Set up signal handler for Ctrl+C
    trap 'cleanup_downloads; echo ""; log_warn "Downloads aborted by user."; read -p "Press Enter to continue..."; return' INT

    # Create directories first
    create_directories

    # Clear screen and set up display
    clear_menu
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    Downloading Datasets${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Press ${YELLOW}Ctrl+C${NC} to abort all downloads"
    echo ""

    # Save cursor position for status display
    DOWNLOAD_START_LINE=8

    # Initialize status display
    local line=0
    for dataset in "${DATASET_ORDER[@]}"; do
        if [[ "${DATASET_SELECTED[$dataset]}" == "1" ]]; then
            write_status "$dataset" "waiting" 0
            update_dataset_status "$dataset" "$line"
            ((line++))
        fi
    done

    # Add extra lines for spacing
    echo ""
    echo ""
    printf "${HIDE_CURSOR}"

    # Start all downloads in parallel
    for dataset in "${DATASET_ORDER[@]}"; do
        if [[ "${DATASET_SELECTED[$dataset]}" == "1" ]]; then
            case "$dataset" in
                mnist)     download_mnist_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
                dsprites)  download_dsprites_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
                emnist)    download_emnist_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
                kmnist)    download_kmnist_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
                cifar10)   download_cifar10_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
                cifar100)  download_cifar100_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
                3dchairs)  download_3dchairs_tracked &
                           DOWNLOAD_PIDS[$dataset]=$! ;;
            esac
        fi
    done

    # Monitor progress until all complete
    local all_done=false
    while [[ "$all_done" == "false" && "$ABORT_DOWNLOADS" == "false" ]]; do
        all_done=true
        line=0

        for dataset in "${DATASET_ORDER[@]}"; do
            if [[ "${DATASET_SELECTED[$dataset]}" == "1" ]]; then
                # Read status from file (format: status:progress:speed)
                local status_line
                status_line=$(read_status "$dataset")
                IFS=':' read -r status progress speed <<< "$status_line"

                DOWNLOAD_STATUS[$dataset]="$status"
                DOWNLOAD_PROGRESS[$dataset]="$progress"
                DOWNLOAD_SPEED[$dataset]="${speed:-0}"

                update_dataset_status "$dataset" "$line"
                ((line++))

                # Check if still running
                if [[ "$status" != "done" && "$status" != "failed" && "$status" != "skipped" ]]; then
                    all_done=false
                fi
            fi
        done

        sleep 0.3
    done

    printf "${SHOW_CURSOR}"

    # Move cursor below status display
    move_to_line "$((line + 1))"
    echo ""

    # Reset signal handler
    trap - INT

    if [[ "$ABORT_DOWNLOADS" == "true" ]]; then
        log_warn "Downloads aborted."
    else
        echo "========================================"
        log_info "All downloads complete!"
        echo "========================================"
    fi

    echo ""
    read -p "Press Enter to continue..."

    # Clean up
    rm -rf "$STATUS_DIR"
    trap - EXIT
}

# Interactive menu
interactive_menu() {
    # Initialize selection
    init_selection

    while true; do
        clear_menu
        print_header
        print_dataset_table
        print_instructions

        local selected_count
        selected_count=$(count_selected)
        echo -e "Selected: ${YELLOW}$selected_count${NC} dataset(s)"
        echo ""

        read -p "Enter command: " cmd

        case "$cmd" in
            [1-7])
                toggle_selection "$cmd"
                ;;
            a|A)
                select_all_missing
                ;;
            n|N)
                deselect_all
                ;;
            d|D)
                download_selected
                # Re-initialize selection after download
                init_selection
                ;;
            q|Q)
                echo ""
                log_info "Aborted by user."
                exit 0
                ;;
            *)
                # Invalid input, just refresh
                ;;
        esac
    done
}

# Main function
main() {
    # Run interactive menu
    interactive_menu
}

# Run main function
main "$@"
