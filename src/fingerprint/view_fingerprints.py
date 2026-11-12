"""
Show every segment in a fingerprint .pkl: an overview with all masks/boxes, then one
panel per object (its mask highlighted, box, label, confidence, mask area).

    python src/fingerprint/view_fingerprints.py fingerprints/test_img.pkl src/fingerprint/test_img.jpg
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def color(i):
    hue = int(i * 180 / 10.0 * 0.618 * 10) % 180  # spread hues
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def label(img, text, x, y, col):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), col, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw(img, mask, box, text, col, alpha):
    out = img.copy()
    out[mask] = (alpha * np.array(col) + (1 - alpha) * out[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, col, 2)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
    label(out, text, max(x1, 0), y1, col)
    return out


def segment_sheet(d, img, cols=4):
    """Contact sheet (BGR) for a DovSG det_res dict `d` made from BGR image `img`:
    overview of all segments, the original, then one panel per segment."""
    h, w = img.shape[:2]
    masks = np.asarray(d["mask"], dtype=bool).reshape(-1, h, w)  # (0, h, w) when nothing was detected

    names = [d["classes"][c] for c in d["class_id"]]
    texts = [f"#{i} {n} {s:.2f}" for i, (n, s) in enumerate(zip(names, d["confidence"]))]

    # overview: largest masks first so small ones stay visible on top
    overview = img.copy()
    for i in np.argsort([-m.sum() for m in masks]):
        overview = draw(overview, masks[i], d["xyxy"][i], texts[i], color(i), 0.35)
    label(overview, f"all {len(names)} segments", 4, h - 4, (255, 255, 255))

    panels = [overview, img.copy()]
    label(panels[1], "original", 4, h - 4, (255, 255, 255))
    for i in range(len(names)):
        dimmed = (img * 0.45).astype(np.uint8)
        dimmed[masks[i]] = img[masks[i]]
        p = draw(dimmed, masks[i], d["xyxy"][i], texts[i], color(i), 0.25)
        label(p, f"mask {int(masks[i].sum())} px ({100 * masks[i].mean():.0f}%)", 4, h - 4, (255, 255, 255))
        panels.append(p)

    while len(panels) % cols:
        panels.append(np.full_like(img, 255))
    pad = lambda p: cv2.copyMakeBorder(p, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # noqa: E731
    rows = [np.hstack([pad(p) for p in panels[r:r + cols]]) for r in range(0, len(panels), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl")
    ap.add_argument("image", help="the original image the .pkl was made from")
    ap.add_argument("--out", help="output .png (default: next to the .pkl, *_segments.png)")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    img = cv2.imread(args.image)
    if len(d["class_id"]):
        h, w = np.asarray(d["mask"]).shape[1:]
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
    sheet = segment_sheet(d, img, args.cols)
    names = d["class_id"]

    out = Path(args.out) if args.out else Path(args.pkl).with_name(Path(args.pkl).stem + "_segments.png")
    cv2.imwrite(str(out), sheet)
    print(f"{len(names)} segments -> {out}")


if __name__ == "__main__":
    main()
