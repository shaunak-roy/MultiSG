# Windows twin of setup-rgbd.sh (CAEN lab PCs: no admin, profile wiped at logout, so this
# installs everything from scratch each session). Run from the MultiSG repository root:
#
#   powershell -ExecutionPolicy Bypass -File .\setup-rgbd.ps1 room_run1
#       name of a run zip on the repo's "data" release (downloaded once to work\data\)
#   powershell -ExecutionPolicy Bypass -File .\setup-rgbd.ps1 C:\Downloads\new_room.zip
#       a run zip (local file or https://...zip)
#   powershell -ExecutionPolicy Bypass -File .\setup-rgbd.ps1 C:\data\room_run1
#       a run folder (or its frames\ folder)
#       -> ViT-L metric depth + lidar/floor calibration for every frame, then a frame check
#   New room: zip the run folder (frames\, frames.csv, scans\, info.txt), add the zip to the
#   "data" release on GitHub, then run this with the zip name (without .zip).
#   powershell -ExecutionPolicy Bypass -File .\setup-rgbd.ps1 C:\data\room_run1.mp4
#       -> uncalibrated ViT-S depth (old video path)
#   ... -Encoder vits      smaller/faster model in frames mode
param(
    [string]$Video = "",
    [ValidateSet("vits", "vitb", "vitl")]
    [string]$Encoder = "vitl"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
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

function Get-File {
    # curl.exe ships with Windows 10+; resumes with -C - if a download was cut off
    param([string]$Url, [string]$Out)
    if (Test-Path $Out) { return }
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        Invoke-Checked "curl.exe" @("-fL", "--retry", "5", "--retry-all-errors", "-C", "-",
                                    "-o", "$Out.download", $Url)
    }
    else {
        Invoke-WebRequest -Uri $Url -OutFile "$Out.download"
    }
    Move-Item "$Out.download" $Out
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
    $Video = Read-Host "Frames folder, video file path or direct download URL"
}

if (-not $Video) {
    throw "No input provided."
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

# Pick the PyTorch build the NVIDIA driver can run ("CUDA Version" in nvidia-smi):
#   >= 13.2 -> cu132, torch 2.15 | >= 12.8 -> cu128, torch 2.11 | >= 12.6 -> cu126, torch 2.14
#   otherwise CPU. (cu128 stops at torch 2.11, which still supports RTX 2000 Ada / RTX 50xx.)
$torchIndex = "https://download.pytorch.org/whl/cpu"
$torchSpec = "torch>=2.15"
$driverCuda = $null

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = (& nvidia-smi 2>$null) -join "`n"
    if ($LASTEXITCODE -eq 0 -and $smi -match "CUDA Version:\s*([0-9]+\.[0-9]+)") {
        $driverCuda = [version]$Matches[1]
    }
}

if ($driverCuda) {
    Write-Host "NVIDIA GPU detected (driver supports CUDA $driverCuda)."
    if ($driverCuda -ge [version]"13.2") {
        $torchIndex = "https://download.pytorch.org/whl/cu132"
    }
    elseif ($driverCuda -ge [version]"12.8") {
        $torchIndex = "https://download.pytorch.org/whl/cu128"; $torchSpec = "torch>=2.7,<2.12"
    }
    elseif ($driverCuda -ge [version]"12.6") {
        $torchIndex = "https://download.pytorch.org/whl/cu126"; $torchSpec = "torch>=2.7,<2.15"
    }
    else {
        Write-Host "Driver too old for current CUDA PyTorch: using CPU."
    }
}
else {
    Write-Host "No NVIDIA GPU detected: installing CPU PyTorch."
}
Write-Host "PyTorch: $torchSpec from $torchIndex"

Invoke-Checked "uv" @(
    "add", $torchSpec, "torchvision",
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

# ---------------------------------------------------------------- run given as a zip
# Bare name (published on the "data" release), a .zip URL, or a local .zip.
# Downloaded/unpacked once into work\data\ (git-ignored).
$dataRelease = "https://github.com/shaunak-roy/MultiSG/releases/download/data"
if (-not (Test-Path -LiteralPath $Video) -and $Video -match '^[A-Za-z0-9_.-]+$' -and
    $Video -notmatch '\.(mp4|mov|avi|mkv|webm)$') {
    $Video = "$dataRelease/$($Video -replace '\.zip$', '').zip"
    Write-Host "Using published run: $Video"
}
if ($Video -match '^https?://.*\.zip$') {
    New-Item -ItemType Directory -Force "work/data" | Out-Null
    $zip = "work/data/$(Split-Path $Video -Leaf)"
    try { Get-File $Video $zip }
    catch {
        throw "Download failed: $Video (is the zip uploaded to the 'data' release, with exactly this name?)"
    }
    $Video = $zip
}
if ((Test-Path -LiteralPath $Video -PathType Leaf) -and $Video -match '\.zip$') {
    $dest = "work/data/$([IO.Path]::GetFileNameWithoutExtension($Video))"
    if (-not (Test-Path $dest)) {
        Write-Host "Unpacking $Video -> $dest"
        if (Test-Path "$dest.tmp") { Remove-Item -Recurse -Force "$dest.tmp" }
        New-Item -ItemType Directory -Force "$dest.tmp" | Out-Null
        Invoke-Checked "uv" @("run", "python", "-m", "zipfile", "-e", $Video, "$dest.tmp")
        Move-Item "$dest.tmp" $dest
    }
    $csv = Get-ChildItem -Path $dest -Filter frames.csv -Recurse -Depth 3 | Select-Object -First 1
    if (-not $csv) { throw "No frames.csv inside ${Video}: not a recorded run zip." }
    $Video = Join-Path $csv.DirectoryName "frames"
}
# a run folder (frames\ + frames.csv inside) works as well as the frames folder itself
if ((Test-Path -LiteralPath "$Video/frames" -PathType Container) -and (Test-Path -LiteralPath "$Video/frames.csv")) {
    $Video = Join-Path $Video "frames"
}

# ---------------------------------------------------------------- frames folder mode
# Recorded run (record_room.py): <run>\frames\, <run>\frames.csv, <run>\scans\.
if (Test-Path -LiteralPath $Video -PathType Container) {
    $frames = (Resolve-Path -LiteralPath $Video).Path
    $runDir = Split-Path $frames -Parent
    if (-not (Test-Path "$runDir/frames.csv") -or -not (Test-Path "$runDir/scans")) {
        throw "Expected $runDir\frames.csv and $runDir\scans\ next to the frames folder."
    }

    $names = @{ vits = "Small"; vitb = "Base"; vitl = "Large" }
    $w = "work/checkpoints/depth_anything_v2_metric_hypersim_$Encoder.pth"
    Get-File "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-$($names[$Encoder])/resolve/main/depth_anything_v2_metric_hypersim_$Encoder.pth" $w

    Invoke-Checked "uv" @("run", "python", "-c", $deviceCheck)

    $out = "outputs/$(Split-Path $runDir -Leaf)_$Encoder"
    Invoke-Checked "uv" @("run", "python", "-m", "multi_sg.run_frames",
                          "--frames", $frames, "--encoder", $Encoder, "--out", $out)
    Invoke-Checked "uv" @("run", "python", "-m", "multi_sg.calibrate",
                          "--run", $runDir, "--depth", "$out/depth_mm", "--apply")

    Write-Host "Raw network depth:   $out/depth_mm/"
    Write-Host "Calibrated depth:    $out/calibrated/depth_mm/   (16-bit mm, same names as the photos)"
    Write-Host "Side-by-side views:  $out/calibrated/preview/"
    Write-Host "Check any frame later:"
    Write-Host "  uv run python -m multi_sg.check_frame --run `"$runDir`""

    # Spot-check any frame (photo + lidar dots, raw depth, calibrated depth).
    Write-Host ""
    $answer = Read-Host "Check a frame now? [y/N]"
    if ($answer -match "^[Yy]") {
        Invoke-Checked "uv" @("run", "python", "-m", "multi_sg.check_frame",
                              "--run", $runDir, "--depth", $out)
    }
    exit 0
}

# ---------------------------------------------------------------- video mode (uncalibrated)
$weights = "work/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
Get-File "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth" $weights

# Download a supplied direct video URL.
if ($Video -match "^https?://") {
    $downloaded = "work/videos/input-$(Get-Date -Format 'yyyyMMdd-HHmmss').video"
    Get-File $Video $downloaded
    $Video = $downloaded
}

if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) {
    throw "Video file or frames folder not found: $Video"
}

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
    "--save-mm",
    "--preview"
)

Write-Host "Results saved to: $output"
