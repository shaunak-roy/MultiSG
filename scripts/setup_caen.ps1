<#
.SYNOPSIS
  One-shot, re-runnable setup of MultiSG + DovSG perception on a CAEN Windows lab PC (no admin).

.DESCRIPTION
  CAEN wipes the local profile (C:\Users\<you>) at logout, so this script rebuilds everything:
    1. git (portable MinGit, if git isn't already on PATH)  -> %LOCALAPPDATA%\Programs\MinGit
    2. uv (pinned)                                          -> %USERPROFILE%\.local\bin
    3. clones the repo (if needed) and the DovSG submodules
    4. project env (.venv, uv sync) and DovSG env (.venv-dovsg, Python 3.10 + CUDA torch)
    5. 11 GB of checkpoints, cached on your persistent N: drive so later rebuilds copy
       them over the campus network instead of re-downloading
    6. a check that torch sees the GPU and every package imports
  Every step is skipped if it is already done, so re-running after a partial failure is cheap.
  The script also copies itself to the cache dir, so after a wipe you can run it from N:.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_caen.ps1
.EXAMPLE
  # after a profile wipe (repo gone): run the backup copy, it clones the repo for you
  powershell -ExecutionPolicy Bypass -File N:\multisg-cache\setup_caen.ps1
#>
param(
    # where the repo lives (cloned here if missing). Local disk: the venv is slow on a network drive.
    [string]$RepoDir = "",
    [string]$RepoUrl = "https://github.com/shaunak-roy/MultiSG.git",
    # persistent checkpoint cache; "" disables it. Defaults to N:\multisg-cache when N: exists.
    [string]$CacheDir = $(if (Test-Path "N:\") { "N:\multisg-cache" } else { "" }),
    [string]$UvVersion = "0.13.7"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Expand-Archive is very slow with the progress bar in PS 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$curl = "$env:SystemRoot\System32\curl.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" }
function Run {
    # run a native command and stop on a non-zero exit code (PowerShell 5.1 doesn't).
    # No param() block on purpose: declared parameters would capture flags like `pip install -e`.
    $exe, $rest = $args
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $exe $($rest -join ' ')" }
}
function Add-UserPath($dir) {
    # current session + user PATH (HKCU, no admin). The user PATH dies with the profile; re-run this script.
    if (($env:Path -split ';') -notcontains $dir) { $env:Path = "$dir;$env:Path" }
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $user) { $user = "" }
    if (($user -split ';') -notcontains $dir) {
        [Environment]::SetEnvironmentVariable("Path", (($dir, $user) -join ';').TrimEnd(';'), "User")
    }
}
function Download($url, $dest) {
    # resumable download to <dest>.part, renamed when complete
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Run $curl -L --fail --retry 3 -C - -o "$dest.part" $url
    Move-Item -Force "$dest.part" $dest
}

# ---------------------------------------------------------------- 1. git
Step "git"
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    foreach ($d in "$env:LOCALAPPDATA\Programs\Git\cmd", "$env:LOCALAPPDATA\Programs\MinGit\cmd") {
        if (Test-Path "$d\git.exe") { Add-UserPath $d; break }
    }
    $git = Get-Command git -ErrorAction SilentlyContinue
}
if (-not $git) {
    $rel = Invoke-RestMethod "https://api.github.com/repos/git-for-windows/git/releases/latest" -Headers @{ "User-Agent" = "setup-script" }
    $asset = $rel.assets | Where-Object { $_.name -match '^MinGit-[\d.]+-64-bit\.zip$' } | Select-Object -First 1
    if (-not $asset) { throw "could not find a MinGit 64-bit zip in the latest git-for-windows release" }
    $zip = Join-Path $env:TEMP $asset.name
    Info "installing $($asset.name) (portable, no admin)"
    Download $asset.browser_download_url $zip
    $dest = "$env:LOCALAPPDATA\Programs\MinGit"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Expand-Archive $zip $dest
    Remove-Item $zip
    Add-UserPath "$dest\cmd"
}
Info (git --version)

# ---------------------------------------------------------------- 2. uv
Step "uv $UvVersion"
$uvDir = "$env:USERPROFILE\.local\bin"
$uv = "$uvDir\uv.exe"
$have = if (Test-Path $uv) { (& $uv --version) -split ' ' | Select-Object -Index 1 } else { "" }
if ($have -ne $UvVersion) {
    $zip = Join-Path $env:TEMP "uv-$UvVersion.zip"
    Download "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $zip
    New-Item -ItemType Directory -Force $uvDir | Out-Null
    Expand-Archive -Force $zip $uvDir
    Remove-Item $zip
}
Add-UserPath $uvDir
Info (& $uv --version)

# ---------------------------------------------------------------- 3. repo + submodules
Step "repository"
if (-not $RepoDir) {
    $here = Split-Path -Parent $PSScriptRoot
    $RepoDir = if (Test-Path (Join-Path $here "pyproject.toml")) { $here } else { "$env:USERPROFILE\MultiSG" }
}
if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    Info "cloning $RepoUrl -> $RepoDir"
    Run git clone $RepoUrl $RepoDir
}
$RepoDir = (Resolve-Path $RepoDir).Path
Set-Location $RepoDir
Info $RepoDir
Run git submodule update --init third_party/DovSG
Run git -C third_party/DovSG submodule update --init third_party/segment-anything-2 third_party/GroundingDINO third_party/recognize-anything
$tp = "$RepoDir\third_party\DovSG\third_party"

# ---------------------------------------------------------------- 4a. project env
Step "project env (.venv via uv sync)"
# On CAEN PCs uv intermittently fails with "Access is denied" removing a wheel's *.data folder
# (security software holding it briefly); a retry picks up where it stopped.
# --no-install-project: pyproject.toml names a src/multi_sg package that doesn't exist (the code is
# plain scripts), so building the project itself fails on a fresh clone. Install the dependencies only.
foreach ($try in 1..3) {
    & $uv sync --no-install-project
    if ($LASTEXITCODE -eq 0) { break }
    if ($try -eq 3) { throw "uv sync failed 3 times" }
    Info "uv sync failed, retrying ($try/3)"
    Start-Sleep -Seconds 3
}

# ---------------------------------------------------------------- 4b. DovSG env
Step "DovSG env (.venv-dovsg, Python 3.10)"
$py = "$RepoDir\.venv-dovsg\Scripts\python.exe"
$check = "import torch, sam2, groundingdino, ram, open_clip, supervision, open3d, imageio_ffmpeg; " +
         "assert torch.__version__ == '2.3.1+cu121' and supervision.__version__ == '0.14.0'; " +
         "print(torch.__version__, 'cuda', torch.cuda.is_available(), 'supervision', supervision.__version__)"
$envOk = $false
if (Test-Path $py) {
    $ErrorActionPreference = "Continue"  # PS 5.1 turns redirected native stderr into errors
    & $py -c $check 2>$null | Out-Null
    $envOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
}
if ($envOk) {
    Info "already installed, skipping"
} else {
    if (-not (Test-Path $py)) { Run $uv venv .venv-dovsg --python 3.10 --seed }
    # pip, not uv pip: uv fails to install intel-openmp (torch dep) on Windows with "Access is denied"
    Run $py -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    # no CUDA toolkit/compiler without admin: SAM2's optional CUDA op is skipped, and GroundingDINO
    # builds without its op (src/fingerprint/fingerprint_image.py switches it to the PyTorch version)
    $env:SAM2_BUILD_CUDA = "0"
    Run $py -m pip install --no-build-isolation -e "$tp\segment-anything-2"
    Run $py -m pip install --no-build-isolation -e "$tp\GroundingDINO"
    Run $py -m pip install -r "$tp\recognize-anything\requirements.txt"
    Run $py -m pip install --no-build-isolation -e "$tp\recognize-anything"
    # must run last (pins over what the above pulled in). GroundingDINO's supervision>=0.22 warning is expected.
    Run $py -m pip install numpy==1.26.4 supervision==0.14.0 open_clip_torch==2.31.0 transformers==4.50.3 open3d scipy matplotlib opencv-python tqdm imageio-ffmpeg
}

# ---------------------------------------------------------------- 5. checkpoints
Step "checkpoints (~11 GB)"
$ck = "$RepoDir\third_party\DovSG\checkpoints"
$hf = "https://huggingface.co"
$files = @(
    # path under checkpoints/, url, exact size in bytes
    @("bert-base-uncased/config.json",           "$hf/google-bert/bert-base-uncased/resolve/main/config.json", 570),
    @("bert-base-uncased/model.safetensors",     "$hf/google-bert/bert-base-uncased/resolve/main/model.safetensors", 440449768),
    @("bert-base-uncased/tokenizer_config.json", "$hf/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json", 48),
    @("bert-base-uncased/tokenizer.json",        "$hf/google-bert/bert-base-uncased/resolve/main/tokenizer.json", 466062),
    @("bert-base-uncased/vocab.txt",             "$hf/google-bert/bert-base-uncased/resolve/main/vocab.txt", 231508),
    @("CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
      "$hf/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin", 3944692325),
    @("recognize_anything/ram_swin_large_14m.pth",
      "$hf/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth", 5625634877),
    @("GroundingDINO/groundingdino_swint_ogc.pth",
      "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", 693997677),
    # Meta's own HF copy: dl.fbaipublicfiles.com served an expired TLS certificate (Nov 2026)
    @("segment-anything-2/sam2_hiera_large.pt",
      "$hf/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt", 897952466)
)
function Has-File($path, $size) { (Test-Path $path) -and ((Get-Item $path).Length -eq $size) }

foreach ($f in $files) {
    $rel, $url, $size = $f
    $dest = Join-Path $ck $rel
    $cached = if ($CacheDir) { Join-Path "$CacheDir\checkpoints" $rel } else { "" }
    if (Has-File $dest $size) {
        Info "ok        $rel"
    } elseif ($cached -and (Has-File $cached $size)) {
        Info "from cache $rel"
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item $cached $dest
    } else {
        Info "download  $rel ($([math]::Round($size / 1GB, 2)) GB)"
        Download $url $dest
        if (-not (Has-File $dest $size)) { throw "$rel has the wrong size after download; delete it and re-run" }
    }
    if ($cached -and -not (Has-File $cached $size)) {
        try {
            New-Item -ItemType Directory -Force (Split-Path $cached) | Out-Null
            Copy-Item $dest $cached
        } catch {
            Write-Warning "could not cache $rel in $CacheDir ($($_.Exception.Message)); it will be re-downloaded next time"
        }
    }
}
Copy-Item -Force "$tp\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py" "$ck\GroundingDINO\"

if ($CacheDir) {
    try {
        Copy-Item -Force $PSCommandPath (Join-Path $CacheDir "setup_caen.ps1")
        Info "backup of this script: $CacheDir\setup_caen.ps1"
    } catch { Write-Warning "could not copy this script to $CacheDir" }
}

# ---------------------------------------------------------------- 6. verify
Step "verify"
Run $py -c $check
$gpu = & $py -c "import torch; print(torch.cuda.is_available())"
if ($gpu -ne "True") { Write-Warning "torch does not see a GPU; use --device cpu" }

Write-Host "`nDone. Try:" -ForegroundColor Green
Write-Host "  cd $RepoDir"
Write-Host "  .venv-dovsg\Scripts\python.exe src\fingerprint\fingerprint_image.py src\fingerprint\test_img.jpg"
Write-Host "  .venv-dovsg\Scripts\python.exe src\fingerprint\live_fingerprint.py --url tcp://<pi-ip>:8888"
Write-Host "Open a new terminal for git/uv to be on PATH there (this session already has them)."
