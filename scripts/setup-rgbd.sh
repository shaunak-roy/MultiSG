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

# Accept a local path or direct video download URL.
VIDEO="${1:-}"

if [[ -z "$VIDEO" ]]; then
    read -r -p "Video file path or direct download URL: " VIDEO
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
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

if [[ "$(uname -s)" == "Darwin" ]]; then
    # Standard macOS wheels support MPS on compatible Macs.
    TORCH_INDEX="https://pypi.org/simple"
elif command -v nvidia-smi >/dev/null &&
     nvidia-smi -L >/dev/null 2>&1; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    echo "NVIDIA GPU detected: installing CUDA 12.8 PyTorch."
else
    echo "No NVIDIA GPU detected: installing CPU PyTorch."
fi

uv add torch torchvision --index "pytorch=$TORCH_INDEX"

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
    --max-frames 10 \
    --save-mm \
    --preview

echo "Results saved to: $OUTPUT"