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

DOVSG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "DovSG"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--out", type=str, default="fingerprints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--add_classes", type=str, nargs="*", default=[])
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint paths in dovsg/utils/utils.py are relative to the DovSG root
    os.chdir(DOVSG_ROOT)
    sys.path.insert(0, str(DOVSG_ROOT))
    from dovsg.memory.ram_groundingdino_sam2_clip_semantic_memory import RamGroundingDinoSAM2ClipDataset

    semantic_memory = RamGroundingDinoSAM2ClipDataset(device=args.device)
    # DovSG hardcodes add_classes for its lab scenes
    semantic_memory.add_classes = args.add_classes

    image = np.array(Image.open(image_path).convert("RGB"))
    with torch.no_grad():
        det_res, annotated_image, image_pil = semantic_memory.semantic_process(image=image)

    name = image_path.stem
    cv2.imwrite(str(out_dir / f"{name}.jpg"), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    with open(out_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(det_res, f)

    for class_id, confidence in zip(det_res["class_id"], det_res["confidence"]):
        print(f"{det_res['classes'][class_id]} {confidence:0.2f}")
    print(f"==> {len(det_res['class_id'])} fingerprints saved to {out_dir / f'{name}.pkl'}")


if __name__ == "__main__":
    main()
