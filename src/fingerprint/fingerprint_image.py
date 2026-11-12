"""
Object fingerprints for a single RGB image.

All perception code is DovSG's, imported unchanged from third_party/DovSG:
    https://github.com/BJHYZJ/DovSG
    Yan et al., "Dynamic Open-Vocabulary 3D Scene Graphs for Long-Term Language-Guided
    Mobile Manipulation", IEEE RA-L 2025, doi:10.1109/LRA.2025.3547643
This script mirrors the semantic memory loop in dovsg/controller.py.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_fingerprints import segment_sheet  # noqa: E402

DOVSG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "DovSG"


def use_pytorch_deform_attn_if_no_cuda_op():
    """Without the compiled groundingdino._C op (no CUDA toolkit / compiler, e.g. on Windows
    without admin rights), Grounding DINO crashes on GPU tensors. Route its CUDA path to the
    pure-PyTorch implementation it ships, which runs on GPU too."""
    from groundingdino.models.GroundingDINO import ms_deform_attn

    if hasattr(ms_deform_attn, "_C"):
        return

    class PyTorchDeformAttn:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return ms_deform_attn.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

    ms_deform_attn.MultiScaleDeformableAttnFunction = PyTorchDeformAttn


def load_semantic_memory(device="cuda", add_classes=()):
    """Load DovSG's RAM + Grounding DINO + SAM2 + CLIP pipeline once.
    Changes the working directory to the DovSG root, so resolve your own paths first."""
    # checkpoint paths in dovsg/utils/utils.py are relative to the DovSG root
    os.chdir(DOVSG_ROOT)
    sys.path.insert(0, str(DOVSG_ROOT))
    use_pytorch_deform_attn_if_no_cuda_op()
    from dovsg.memory.ram_groundingdino_sam2_clip_semantic_memory import RamGroundingDinoSAM2ClipDataset

    semantic_memory = RamGroundingDinoSAM2ClipDataset(device=device)
    # DovSG hardcodes add_classes for its lab scenes
    semantic_memory.add_classes = list(add_classes)
    return semantic_memory


def save_segment_sheet(det_res, image_rgb, path):
    """Save an overview + one panel per segment (mask, box, label, confidence) to `path`."""
    cv2.imwrite(str(path), segment_sheet(det_res, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--out", type=str, default="fingerprints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--add_classes", type=str, nargs="*", default=[])
    parser.add_argument("--no_segments", action="store_true", help="skip the per-segment <name>_segments.png sheet")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_memory = load_semantic_memory(args.device, args.add_classes)

    image = np.array(Image.open(image_path).convert("RGB"))
    with torch.no_grad():
        det_res, annotated_image, image_pil = semantic_memory.semantic_process(image=image)

    name = image_path.stem
    cv2.imwrite(str(out_dir / f"{name}.jpg"), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    with open(out_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(det_res, f)
    if not args.no_segments:
        save_segment_sheet(det_res, image, out_dir / f"{name}_segments.png")

    for class_id, confidence in zip(det_res["class_id"], det_res["confidence"]):
        print(f"{det_res['classes'][class_id]} {confidence:0.2f}")
    print(f"==> {len(det_res['class_id'])} fingerprints saved to {out_dir / f'{name}.pkl'}")


if __name__ == "__main__":
    main()
