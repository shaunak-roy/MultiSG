"""Run Depth Anything V2 (metric) on a folder of recorded frames, e.g. room_run1/frames.

Writes <out>/depth_mm/NNNNNN.png (16-bit millimetres, same file stem as the photo, 0 = invalid)
plus run.json. Resumable: frames whose output already exists are skipped.

  uv run python -m multi_sg.run_frames                       # all frames, ViT-S
  uv run python -m multi_sg.run_frames --every 5             # every 5th frame

View:  uv run python src/multi_sg/view_depth.py --rgb-dir ~/rob535/data/room_run1/frames \
           --depth-dir outputs/room_run1_vits/depth_mm
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from multi_sg.rgbd import depth_millimeters

REPO = Path(__file__).resolve().parents[2]


class RelativePredictor:
    """Depth Anything V2 *relative* model. Its output is disparity (inverse depth) up to an
    unknown scale and shift. We store pseudo-depth = median(disp) / disp (median ~1 'm'),
    which is NOT metric: calibrate.py's per-frame fit maps it to metres."""

    def __init__(self, repo, checkpoint, encoder="vitl", device="auto", input_size=518):
        import sys
        import torch
        from multi_sg.metric_depth import CONFIGS
        sys.path.insert(0, str(Path(repo).resolve()))
        from depth_anything_v2.dpt import DepthAnythingV2
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = DepthAnythingV2(encoder=encoder, **CONFIGS[encoder])
        self.m.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.m.to(device).eval()
        self.input_size = input_size
        self.metadata = dict(model="Depth Anything V2 relative (pseudo-depth = median(disp)/disp)",
                             encoder=encoder, device=device, input_size=input_size,
                             checkpoint=Path(checkpoint).name)

    def predict(self, bgr):
        import numpy as np
        disp = self.m.infer_image(bgr, self.input_size).astype(np.float64)
        disp = np.maximum(disp, 1e-6)
        return np.median(disp) / disp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, default=Path("~/rob535/data/room_run1/frames").expanduser())
    ap.add_argument("--out", type=Path, default=None, help="default outputs/room_run1_<encoder>")
    ap.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vits")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--model-repo", type=Path, default=REPO / "work" / "Depth-Anything-V2")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--relative", action="store_true", help="relative model (not metric)")
    a = ap.parse_args()

    ck = REPO / "work" / "checkpoints"
    if a.relative:
        ckpt = a.checkpoint or ck / f"depth_anything_v2_{a.encoder}.pth"
        out = a.out or REPO / "outputs" / f"{a.frames.parent.name}_rel_{a.encoder}"
        pred = RelativePredictor(a.model_repo, ckpt, a.encoder, a.device, a.input_size)
    else:
        ckpt = a.checkpoint or ck / f"depth_anything_v2_metric_hypersim_{a.encoder}.pth"
        out = a.out or REPO / "outputs" / f"{a.frames.parent.name}_{a.encoder}"
        from multi_sg.metric_depth import MetricDepthPredictor
        pred = MetricDepthPredictor(repo=a.model_repo, checkpoint=ckpt, encoder=a.encoder,
                                    dataset="hypersim", device=a.device, input_size=a.input_size)
    (out / "depth_mm").mkdir(parents=True, exist_ok=True)
    meta = dict(frames=str(a.frames.resolve()), every=a.every, model=pred.metadata,
                depth_png="uint16 millimetres, 0 = invalid, camera optical-axis Z",
                status="running")
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    files = sorted(a.frames.glob("*.jpg"))[::a.every]
    todo = [f for f in files if not (out / "depth_mm" / f"{f.stem}.png").exists()]
    print(f"{len(files)} frames, {len(files) - len(todo)} already done, device {pred.metadata['device']}",
          flush=True)
    t0 = time.time()
    for k, f in enumerate(todo, 1):
        d = pred.predict(cv2.imread(str(f), cv2.IMREAD_COLOR))
        if not cv2.imwrite(str(out / "depth_mm" / f"{f.stem}.png"), depth_millimeters(d)):
            raise OSError(f"could not write depth for {f}")
        if k % 50 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"{k}/{len(todo)}  {el / k:.2f} s/frame  ETA {(len(todo) - k) * el / k / 60:.1f} min",
                  flush=True)
    meta["status"] = "complete"
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done:", out)


if __name__ == "__main__":
    main()
