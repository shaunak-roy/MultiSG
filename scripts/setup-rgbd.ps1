param(
    [string]$Video = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param(
        [string]$Program,
        [string[]]$Arguments
    )

    & $Program @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Install Git first: https://git-scm.com/downloads"
}

if (-not (Test-Path "pyproject.toml")) {
    throw "Save this script in the MultiSG repository root."
}

if (
    -not (Test-Path "src/multi_sg/rgbd.py") -or
    -not (Test-Path "src/multi_sg/metric_depth.py")
) {
    throw "Copy rgbd.py and metric_depth.py into src/multi_sg/ first."
}

if (-not $Video) {
    $Video = Read-Host "Video file path or direct download URL"
}

if (-not $Video) {
    throw "No video provided."
}

# Install uv if needed.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 |
        Invoke-Expression

    $env:Path = "$HOME\.local\bin;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Restart PowerShell so uv is on PATH, then retry."
}

Invoke-Checked "uv" @("python", "install", "3.12")

Invoke-Checked "uv" @(
    "add", "--python", "3.12",
    "numpy", "av", "opencv-python"
)

# Detect an NVIDIA GPU.
$hasNvidia = $false

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi -L *> $null
    $hasNvidia = ($LASTEXITCODE -eq 0)
}

$torchIndex = "https://download.pytorch.org/whl/cpu"

if ($hasNvidia) {
    $torchIndex = "https://download.pytorch.org/whl/cu128"
    Write-Host "NVIDIA GPU detected: installing CUDA 12.8 PyTorch."
}
else {
    Write-Host "No NVIDIA GPU detected: installing CPU PyTorch."
}

Invoke-Checked "uv" @(
    "add", "torch", "torchvision",
    "--index", "pytorch=$torchIndex"
)

New-Item -ItemType Directory -Force "work/checkpoints" | Out-Null
New-Item -ItemType Directory -Force "work/videos" | Out-Null

$modelRepo = "work/Depth-Anything-V2"
$revision = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"

if (-not (Test-Path $modelRepo)) {
    Invoke-Checked "git" @(
        "clone",
        "https://github.com/DepthAnything/Depth-Anything-V2.git",
        $modelRepo
    )

    Invoke-Checked "git" @(
        "-C", $modelRepo, "checkout", $revision
    )
}

if (-not (Test-Path "$modelRepo/metric_depth/depth_anything_v2/dpt.py")) {
    throw "Missing the official metric_depth implementation."
}

$weights = "work/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
$weightsUrl = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth"

if (-not (Test-Path $weights)) {
    Invoke-WebRequest -Uri $weightsUrl -OutFile "$weights.download"
    Move-Item "$weights.download" $weights
}

# Download a supplied direct video URL.
if ($Video -match "^https?://") {
    $downloaded = "work/videos/input-$(Get-Date -Format 'yyyyMMdd-HHmmss').video"

    Invoke-WebRequest -Uri $Video -OutFile "$downloaded.download"
    Move-Item "$downloaded.download" $downloaded

    $Video = $downloaded
}

if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) {
    throw "Video file not found: $Video"
}

# Report the backend PyTorch can actually use.
$deviceCheck = @"
import torch
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("Using Apple MPS GPU")
else:
    print("Using CPU; no compatible GPU backend is available")
"@

Invoke-Checked "uv" @("run", "python", "-c", $deviceCheck)

$output = "outputs/test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

Invoke-Checked "uv" @(
    "run", "python", "-m", "multi_sg.rgbd",
    "--video", $Video,
    "--output", $output,
    "--model-repo", $modelRepo,
    "--checkpoint", $weights,
    "--encoder", "vits",
    "--dataset", "hypersim",
    "--device", "auto",
    "--max-frames", "10",
    "--save-mm",
    "--preview"
)

Write-Host "Results saved to: $output"