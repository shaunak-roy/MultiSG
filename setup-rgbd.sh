#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

command -v git >/dev/null || {
    echo "Install Git first."
    exit 1
}

command -v curl >/dev/null || {
    echo "Install curl first."
    exit 1
}

[[ -f pyproject.toml ]] || {
    echo "Save this script in the MultiSG repository root."
    exit 1
}

[[ -f src/multi_sg/rgbd.py && -f src/multi_sg/metric_depth.py ]] || {
    echo "Copy rgbd.py and metric_depth.py into src/multi_sg/ first."
    exit 1
}

# Accepts a recorded run (record_room.py: <run>/frames/, frames.csv, scans/) in any form:
#   ./setup-rgbd.sh room_run1                  name of a run zip on the repo's "data" release
#                                              (downloaded once to work/data/)
#   ./setup-rgbd.sh ~/Downloads/new_room.zip   a run zip (local file or https://...zip)
#   ./setup-rgbd.sh ~/rob535/data/room_run1    a run folder (or its frames/ folder)
#     -> ViT-L metric depth + lidar/floor calibration for every frame, then a frame check
#   ./setup-rgbd.sh room_run1.mp4              a video -> uncalibrated ViT-S depth (old path)
# New room: zip the run folder (frames/, frames.csv, scans/, info.txt), add the zip to the
# "data" release on GitHub, then ./setup-rgbd.sh <zip name without .zip>.
VIDEO="${1:-}"
ENCODER="${ENCODER:-vitl}"   # frames mode: vits | vitb | vitl

if [[ -z "$VIDEO" ]]; then
    read -r -p "Frames folder, video file path or direct download URL: " VIDEO
fi

[[ -n "$VIDEO" ]] || {
    echo "No video provided."
    exit 1
}

# Install uv if needed.
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

command -v uv >/dev/null || {
    echo "Restart your terminal so uv is on PATH, then retry."
    exit 1
}

uv python install 3.12
uv add --python 3.12 numpy av opencv-python

# Choose the PyTorch package source.
# The build must match what the NVIDIA driver supports ("CUDA Version" in nvidia-smi):
#   driver CUDA >= 13.2 -> cu132, torch 2.15  (e.g. OMEN laptop, driver 595)
#   driver CUDA >= 12.8 -> cu128, torch 2.11  (cu128 stops at 2.11; still has sm_89/sm_120)
#   driver CUDA >= 12.6 -> cu126, torch 2.14
#   otherwise           -> CPU
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_SPEC="torch>=2.15"

ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }

if [[ "$(uname -s)" == "Darwin" ]]; then
    # Standard macOS wheels support MPS on compatible Macs.
    TORCH_INDEX="https://pypi.org/simple"
elif command -v nvidia-smi >/dev/null &&
     nvidia-smi -L >/dev/null 2>&1; then
    DRIVER_CUDA="$(nvidia-smi | grep -oE 'CUDA Version: *[0-9.]+' | grep -oE '[0-9.]+$' || true)"
    echo "NVIDIA GPU detected: $(nvidia-smi -L | head -1)  (driver supports CUDA ${DRIVER_CUDA:-?})"
    if [[ -n "$DRIVER_CUDA" ]] && ver_ge "$DRIVER_CUDA" 13.2; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu132"
    elif [[ -n "$DRIVER_CUDA" ]] && ver_ge "$DRIVER_CUDA" 12.8; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"; TORCH_SPEC="torch>=2.7,<2.12"
    elif [[ -n "$DRIVER_CUDA" ]] && ver_ge "$DRIVER_CUDA" 12.6; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"; TORCH_SPEC="torch>=2.7,<2.15"
    else
        echo "Driver too old for current CUDA PyTorch: using CPU."
    fi
else
    echo "No NVIDIA GPU detected: installing CPU PyTorch."
fi
echo "PyTorch: $TORCH_SPEC from $TORCH_INDEX"

uv add "$TORCH_SPEC" torchvision --index "pytorch=$TORCH_INDEX"

mkdir -p work/checkpoints work/videos

MODEL_REPO="work/Depth-Anything-V2"
REVISION="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"

if [[ ! -d "$MODEL_REPO" ]]; then
    git clone \
        https://github.com/DepthAnything/Depth-Anything-V2.git \
        "$MODEL_REPO"

    git -C "$MODEL_REPO" checkout "$REVISION"
fi

[[ -f "$MODEL_REPO/metric_depth/depth_anything_v2/dpt.py" ]] || {
    echo "Missing the official metric_depth implementation."
    exit 1
}

WEIGHTS="work/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
WEIGHTS_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth"

if [[ ! -f "$WEIGHTS" ]]; then
    curl -fL --retry 3 "$WEIGHTS_URL" -o "${WEIGHTS}.download"
    mv "${WEIGHTS}.download" "$WEIGHTS"
fi

# ---- recorded run given as a zip: a bare name (published on the "data" release), a .zip
# URL, or a local .zip. Downloaded/unpacked once into work/data/ (git-ignored).
DATA_RELEASE="https://github.com/shaunak-roy/MultiSG/releases/download/data"
if [[ ! -e "$VIDEO" && "$VIDEO" =~ ^[A-Za-z0-9_.-]+$ && ! "$VIDEO" =~ \.(mp4|mov|avi|mkv|webm)$ ]]; then
    VIDEO="$DATA_RELEASE/${VIDEO%.zip}.zip"
    echo "Using published run: $VIDEO"
fi
if [[ "$VIDEO" =~ ^https?://.*\.zip$ ]]; then
    mkdir -p work/data
    ZIP="work/data/$(basename "$VIDEO")"
    if [[ ! -f "$ZIP" ]]; then
        curl -fL --retry 5 --retry-all-errors -C - "$VIDEO" -o "${ZIP}.download" || {
            echo "Download failed: $VIDEO"
            echo "(is the zip uploaded to the 'data' release, with exactly this name?)"
            exit 1
        }
        mv "${ZIP}.download" "$ZIP"
    fi
    VIDEO="$ZIP"
fi
if [[ -f "$VIDEO" && "$VIDEO" == *.zip ]]; then
    DEST="work/data/$(basename "${VIDEO%.zip}")"
    if [[ ! -d "$DEST" ]]; then
        echo "Unpacking $VIDEO -> $DEST"
        rm -rf "${DEST}.tmp" && mkdir -p "${DEST}.tmp"
        uv run python -m zipfile -e "$VIDEO" "${DEST}.tmp"
        mv "${DEST}.tmp" "$DEST"
    fi
    CSV="$(find "$DEST" -maxdepth 3 -name frames.csv | head -1)"
    [[ -n "$CSV" ]] || { echo "No frames.csv inside $VIDEO: not a recorded run zip."; exit 1; }
    VIDEO="$(dirname "$CSV")/frames"
fi
# a run folder (frames/ + frames.csv inside) works as well as the frames folder itself
if [[ -d "$VIDEO/frames" && -f "$VIDEO/frames.csv" ]]; then
    VIDEO="$VIDEO/frames"
fi

# Frames folder: metric depth on every frame, then per-frame calibration with the run's
# lidar scans + odometry (frames.csv, scans/) and config/mbot_mount.yaml.
if [[ -d "$VIDEO" ]]; then
    FRAMES="$(cd "$VIDEO" && pwd)"
    RUN_DIR="$(dirname "$FRAMES")"
    [[ -f "$RUN_DIR/frames.csv" && -d "$RUN_DIR/scans" ]] || {
        echo "Expected $RUN_DIR/frames.csv and $RUN_DIR/scans/ next to the frames folder."
        exit 1
    }
    W="work/checkpoints/depth_anything_v2_metric_hypersim_${ENCODER}.pth"
    if [[ ! -f "$W" ]]; then
        case "$ENCODER" in vits) N=Small ;; vitb) N=Base ;; vitl) N=Large ;; esac
        curl -fL --retry 5 --retry-all-errors -C - \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-${N}/resolve/main/$(basename "$W")" \
            -o "${W}.download"
        mv "${W}.download" "$W"
    fi
    uv run python -c "import torch; print('Using GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('Using CPU (slow: ~10x)')"

    OUT="outputs/$(basename "$RUN_DIR")_${ENCODER}"
    uv run python -m multi_sg.run_frames --frames "$FRAMES" --encoder "$ENCODER" --out "$OUT"
    uv run python -m multi_sg.calibrate --run "$RUN_DIR" --depth "$OUT/depth_mm" --apply
    echo "Raw network depth:   $OUT/depth_mm/"
    echo "Calibrated depth:    $OUT/calibrated/depth_mm/   (16-bit mm, same names as the photos)"
    echo "Side-by-side views:  $OUT/calibrated/preview/"
    echo "Check any frame later:"
    echo "  uv run python -m multi_sg.check_frame --run \"$RUN_DIR\""

    # Spot-check: open any frame (photo + lidar dots, raw depth, calibrated depth).
    # Skipped without a terminal or a display (e.g. over ssh).
    # Later: uv run python -m multi_sg.check_frame --run "$RUN_DIR"
    if [[ -t 0 && ( -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" || "$(uname -s)" == "Darwin" ) ]]; then
        echo
        read -r -p "Check a frame now? [y/N] " ANSWER
        if [[ "$ANSWER" =~ ^[Yy] ]]; then
            uv run python -m multi_sg.check_frame --run "$RUN_DIR" --depth "$OUT"
        fi
    fi
    exit 0
fi

# Download a supplied direct video URL.
if [[ "$VIDEO" =~ ^https?:// ]]; then
    DOWNLOADED="work/videos/input-$(date +%Y%m%d-%H%M%S).video"

    curl -fL --retry 3 "$VIDEO" -o "${DOWNLOADED}.download"
    mv "${DOWNLOADED}.download" "$DOWNLOADED"

    VIDEO="$DOWNLOADED"
fi

[[ -f "$VIDEO" ]] || {
    echo "Video file not found: $VIDEO"
    exit 1
}

# Report the backend PyTorch can actually use.
uv run python - <<'PY'
import torch

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("Using Apple MPS GPU")
else:
    print("Using CPU; no compatible GPU backend is available")
PY

OUTPUT="outputs/test-$(date +%Y%m%d-%H%M%S)"

uv run python -m multi_sg.rgbd \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    --model-repo "$MODEL_REPO" \
    --checkpoint "$WEIGHTS" \
    --encoder vits \
    --dataset hypersim \
    --device auto \
    --save-mm \
    --preview

echo "Results saved to: $OUTPUT"