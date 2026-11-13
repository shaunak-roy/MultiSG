#!/usr/bin/env python3
"""Interactive depth viewer: hover over a frame to read the distance at each pixel.

Shows the photo and the colour-coded depth map side by side. Moving the mouse over
either one shows, for the pixel under the cursor:
  - pixel (u, v)
  - depth Z in metres (distance along the camera's forward axis)
  - straight-line range sqrt(X^2+Y^2+Z^2) in metres
  - the 3D point (X right, Y down, Z forward) in the camera frame, from the intrinsics

Works with:
  setup-rgbd.sh output   outputs/test-.../rgb/*.png + depth_mm/*.png (or depth_m/)
  depth_cloud.py output  --rgb-dir pointing at room_run1/frames, --depth-dir at <out>/depth

Usage:
  python view_depth.py outputs/test-20261112-215807
  python view_depth.py outputs/test-20261112-215807 --start 5
  python view_depth.py --rgb-dir ~/rob535/data/room_run1/frames --depth-dir cloud_out/depth

Controls:
  mouse over image   read depth
  left click         pin a point (label stays on screen)
  right click        clear pins
  <- / ->            previous / next frame   (Page Up / Page Down = 10 frames)
  o                  toggle depth overlay on the photo
  c                  cycle colour scale: auto (2-98 percentile) / fixed 0-vmax
  s                  save a screenshot PNG next to the run folder
  q                  quit

Ground-truth mode (plan step 3), turned on with --gt [CSV]:
  left click         choose the pixel you measured (yellow ring)
  type in the box    the tape distance, then Enter -> one row appended to the CSV
                       "1.02"            straight-line distance lens -> point (what a tape gives)
                       "z1.02"           depth Z along the camera axis instead
                       "1.02 tag box"    anything after the number is saved as a note
  u                  delete the last GT row of the current frame
  Saved points show as green rings labelled true / network depth.
  CSV columns: frame,u,v,true_Z_m,true_range_m,net_Z_m,depth_src,note
  'frame' = the number in the file name (000123.jpg / 00000123.png -> 123), which is the
  room_run1 index for setup-rgbd.sh output made with stride 1.

Depth files: 16-bit PNG in millimetres (default; both scripts write this), float
TIFF/EXR in metres, or .npy in metres. A uint16 PNG is assumed to be mm unless
--depth-scale says otherwise.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# our MBot camera calibration (640x480)
FX, FY, CX, CY = 554.3148, 555.9327, 312.1483, 264.0024
IMG_EXT = (".png", ".jpg", ".jpeg")
DEFAULT_GT = Path("~/rob535/data/room_run1/gt.csv").expanduser()
GT_COLS = ["frame", "u", "v", "true_Z_m", "true_range_m", "net_Z_m", "depth_src", "note"]
DEPTH_EXT = (".png", ".tif", ".tiff", ".exr", ".npy")


def list_files(d, exts):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in exts)


def load_depth(path, scale):
    """Return depth in metres as float32, with 0/invalid -> NaN."""
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
        s = 1.0 if scale is None else scale
    else:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise IOError(f"could not read {path}")
        if d.ndim == 3:
            d = d[..., 0]
        if scale is not None:
            s = scale
        elif d.dtype == np.uint16:
            s = 0.001              # millimetres -> metres
        elif d.dtype == np.uint8:
            sys.exit(f"{path} is 8-bit: that's a preview image, not real depth. "
                     "Point --depth-dir at depth_mm/ or depth_m/.")
        else:
            s = 1.0                # float images are already metres
        d = d.astype(np.float32)
    d = d * s
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def pair_frames(rgb_dir, depth_dir):
    """Match photos and depth maps by file stem (000123.jpg <-> 000123.png)."""
    depth = {p.stem: p for p in list_files(depth_dir, DEPTH_EXT)}
    pairs = [(r, depth[r.stem]) for r in list_files(rgb_dir, IMG_EXT) if r.stem in depth]
    if not pairs:
        sys.exit(f"no matching file names between {rgb_dir} and {depth_dir}")
    return pairs


def pick_dirs(a):
    if a.rgb_dir and a.depth_dir:
        return Path(a.rgb_dir).expanduser(), Path(a.depth_dir).expanduser()
    if not a.run:
        sys.exit("give a run folder, or both --rgb-dir and --depth-dir")
    run = Path(a.run).expanduser()
    rgb = Path(a.rgb_dir).expanduser() if a.rgb_dir else next(
        (run / n for n in ("rgb", "frames", "color") if (run / n).is_dir()), None)
    dep = Path(a.depth_dir).expanduser() if a.depth_dir else next(
        (run / n for n in ("depth_mm", "depth_m", "depth") if (run / n).is_dir()), None)
    if rgb is None or dep is None:
        sys.exit(f"couldn't find rgb/ and depth_mm/ (or depth_m/, depth/) inside {run}")
    return rgb, dep


class GTStore:
    """Append-only CSV of tape-measured distances at clicked pixels."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.rows = []
        if self.path.exists():
            with open(self.path) as fh:
                self.rows = list(csv.DictReader(fh))
        print(f"ground truth: {self.path}  ({len(self.rows)} rows so far)")

    def for_frame(self, frame):
        return [r for r in self.rows if int(r["frame"]) == frame]

    def add(self, row):
        self.rows.append({k: str(row.get(k, "")) for k in GT_COLS})
        self.save()

    def pop_frame(self, frame):
        for k in range(len(self.rows) - 1, -1, -1):
            if int(self.rows[k]["frame"]) == frame:
                r = self.rows.pop(k)
                self.save()
                return r
        return None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=GT_COLS)
            w.writeheader()
            w.writerows(self.rows)
        tmp.replace(self.path)


def frame_id(path):
    try:
        return int(path.stem)
    except ValueError:
        return -1


class Viewer:
    def __init__(self, pairs, a):
        self.pairs, self.a, self.k = pairs, a, min(max(a.start, 0), len(pairs) - 1)
        self.fixed_scale, self.overlay, self.pins = False, False, []
        self.gt = GTStore(a.gt) if a.gt else None
        self.pending = None                      # pixel waiting for a typed distance
        self.fig, (self.ax_rgb, self.ax_d) = plt.subplots(1, 2, figsize=(14, 6.4 if self.gt else 5.8))
        self.fig.subplots_adjust(left=0.02, right=0.93, top=0.86,
                                 bottom=0.12 if self.gt else 0.04, wspace=0.04)
        for ax in (self.ax_rgb, self.ax_d):
            ax.set_xticks([]), ax.set_yticks([])
        self.im_rgb = self.ax_rgb.imshow(np.zeros((2, 2, 3), np.uint8))
        self.im_ov = self.ax_rgb.imshow(np.zeros((2, 2)), cmap=a.cmap, alpha=0.0)
        self.im_d = self.ax_d.imshow(np.zeros((2, 2)), cmap=a.cmap)
        self.im_d.cmap.set_bad("black")
        cax = self.fig.add_axes([0.94, 0.1, 0.012, 0.7])
        self.cbar = self.fig.colorbar(self.im_d, cax=cax)
        self.cbar.set_label("depth Z (m)")
        # crosshair markers on both panels
        self.marks = [ax.plot([], [], "+", ms=18, mew=1.6, color="white")[0]
                      for ax in (self.ax_rgb, self.ax_d)]
        self.readout = self.fig.text(0.02, 0.93, "", family="monospace", fontsize=11)
        self.title = self.fig.text(0.02, 0.975, "", fontsize=10, color="0.35")
        self.pin_artists = []
        self.gt_artists = []
        if self.gt:
            bax = self.fig.add_axes([0.30, 0.025, 0.25, 0.05])
            self.box = TextBox(bax, "tape distance (m)  ", initial="")
            self.box.on_submit(self.on_submit)
            self.gt_msg = self.fig.text(0.57, 0.04, "click a pixel, then type its measured distance",
                                        fontsize=10, color="0.25")
        c = self.fig.canvas
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_press_event", self.on_click)
        c.mpl_connect("key_press_event", self.on_key)
        self.load()

    # ---------- data ----------
    def load(self):
        rgb_p, d_p = self.pairs[self.k]
        img = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)[..., ::-1]
        d = load_depth(d_p, self.a.depth_scale)
        if d.shape != img.shape[:2]:   # depth at a different resolution -> match the photo
            d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.img, self.d = img, d
        H, W = d.shape
        # intrinsics are for 640x480; rescale if the images are a different size
        sx, sy = W / 640.0, H / 480.0
        self.K = (FX * sx, FY * sy, CX * sx, CY * sy)
        self.pins = []
        self.pending = None
        self.redraw(full=True)

    def clim(self):
        if self.fixed_scale:
            return 0.0, self.a.vmax
        v = self.d[np.isfinite(self.d)]
        if v.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(v, [2, 98])
        return float(lo), float(max(hi, lo + 1e-3))

    def point(self, u, v):
        z = self.d[v, u]
        fx, fy, cx, cy = self.K
        return z, (u - cx) * z / fx, (v - cy) * z / fy

    # ---------- drawing ----------
    def redraw(self, full=False):
        H, W = self.d.shape
        lo, hi = self.clim()
        self.im_rgb.set_data(self.img)
        self.im_d.set_data(self.d)
        self.im_ov.set_data(self.d)
        for im in (self.im_rgb, self.im_d, self.im_ov):
            im.set_extent((-0.5, W - 0.5, H - 0.5, -0.5))
        self.im_d.set_clim(lo, hi)
        self.im_ov.set_clim(lo, hi)
        self.im_ov.set_alpha(0.55 if self.overlay else 0.0)
        for ax in (self.ax_rgb, self.ax_d):
            ax.set_xlim(-0.5, W - 0.5), ax.set_ylim(H - 0.5, -0.5)
        v = self.d[np.isfinite(self.d)]
        stats = (f"valid {100 * v.size / self.d.size:.0f}%  min {v.min():.2f}  "
                 f"median {np.median(v):.2f}  max {v.max():.2f} m") if v.size else "no valid depth"
        rgb_p, d_p = self.pairs[self.k]
        scale = f"fixed 0-{self.a.vmax:g} m" if self.fixed_scale else "auto 2-98%"
        self.title.set_text(f"[{self.k + 1}/{len(self.pairs)}]  {rgb_p.name}  |  {d_p.parent.name}/{d_p.name}"
                            f"  |  {stats}  |  colour: {scale}")
        self.draw_pins()
        if full:
            self.readout.set_text("hover over either image")
        self.fig.canvas.draw_idle()

    def draw_pins(self):
        for a in self.pin_artists:
            a.remove()
        self.pin_artists = []
        self.draw_gt()
        for u, v in self.pins:
            z, x, y = self.point(u, v)
            txt = "no depth" if not np.isfinite(z) else f"{z:.2f} m"
            for ax in (self.ax_rgb, self.ax_d):
                self.pin_artists.append(ax.plot(u, v, "o", ms=6, mfc="none", mec="white", mew=1.5)[0])
                self.pin_artists.append(ax.annotate(
                    txt, (u, v), xytext=(6, -6), textcoords="offset points", color="white",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0)))

    def draw_gt(self):
        for a in self.gt_artists:
            a.remove()
        self.gt_artists = []
        if not self.gt:
            return
        frame = frame_id(self.pairs[self.k][0])
        for ax in (self.ax_rgb, self.ax_d):
            for r in self.gt.for_frame(frame):
                u, v = int(r["u"]), int(r["v"])
                z, _, _ = self.point(u, v)
                net = f"{z:.2f}" if np.isfinite(z) else "--"
                self.gt_artists.append(ax.plot(u, v, "o", ms=8, mfc="none", mec="lime", mew=2)[0])
                self.gt_artists.append(ax.annotate(
                    f"true {float(r['true_Z_m']):.2f} / net {net}", (u, v), xytext=(7, 7),
                    textcoords="offset points", color="lime", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65, lw=0)))
            if self.pending is not None:
                self.gt_artists.append(ax.plot(*self.pending, "o", ms=10, mfc="none",
                                               mec="yellow", mew=2)[0])

    def on_submit(self, text):
        text = text.strip()
        if not text:
            return
        if self.pending is None:
            self.gt_msg.set_text("click the measured pixel first")
            self.fig.canvas.draw_idle()
            return
        num, _, note = text.partition(" ")
        is_z = num[:1].lower() == "z"
        try:
            val = float(num[1:] if is_z else num)
        except ValueError:
            self.gt_msg.set_text(f"not a number: {num!r}")
            self.fig.canvas.draw_idle()
            return
        u, v = self.pending
        fx, fy, cx, cy = self.K
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        norm = float(np.linalg.norm(ray))           # range = Z * |ray|
        Z, rng = (val, val * norm) if is_z else (val / norm, val)
        net, _, _ = self.point(u, v)
        rgb_p, d_p = self.pairs[self.k]
        self.gt.add(dict(frame=frame_id(rgb_p), u=u, v=v, true_Z_m=f"{Z:.4f}",
                         true_range_m=f"{rng:.4f}",
                         net_Z_m=f"{net:.4f}" if np.isfinite(net) else "",
                         depth_src=d_p.parent.parent.name, note=note.strip()))
        err = f"   network {net:.2f} m ({100 * (net - Z) / Z:+.0f}%)" if np.isfinite(net) else ""
        self.gt_msg.set_text(f"saved frame {frame_id(rgb_p)} ({u},{v}): Z = {Z:.3f} m{err}"
                             f"   [{len(self.gt.rows)} rows]")
        print(self.gt_msg.get_text())
        self.pending = None
        self.box.set_val("")
        self.draw_gt()
        self.fig.canvas.draw_idle()

    def pixel(self, e):
        if e.inaxes not in (self.ax_rgb, self.ax_d) or e.xdata is None:
            return None
        H, W = self.d.shape
        u, v = int(round(e.xdata)), int(round(e.ydata))
        return (u, v) if 0 <= u < W and 0 <= v < H else None

    # ---------- events ----------
    def on_move(self, e):
        p = self.pixel(e)
        if p is None:
            return
        u, v = p
        z, x, y = self.point(u, v)
        for m in self.marks:
            m.set_data([u], [v])
        if np.isfinite(z):
            r = float(np.sqrt(x * x + y * y + z * z))
            r_g, g_, b_ = self.img[v, u]
            self.readout.set_text(
                f"pixel (u={u:3d}, v={v:3d})   depth Z = {z:5.2f} m   range = {r:5.2f} m   "
                f"X = {x:+.2f}  Y = {y:+.2f} m   rgb=({r_g},{g_},{b_})")
        else:
            self.readout.set_text(f"pixel (u={u:3d}, v={v:3d})   no depth here")
        self.fig.canvas.draw_idle()

    def on_click(self, e):
        p = self.pixel(e)
        if e.button == 3:
            self.pins = []
        elif e.button == 1 and p is not None:
            if self.gt:
                self.pending = p
                self.gt_msg.set_text(f"pixel ({p[0]},{p[1]}) selected: type its distance, Enter")
            else:
                self.pins.append(p)
        else:
            return
        self.draw_pins()
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        if self.gt and self.box.capturekeystrokes:   # typing in the box, not a command
            return
        if self.gt and e.key == "u":
            r = self.gt.pop_frame(frame_id(self.pairs[self.k][0]))
            self.gt_msg.set_text(f"removed ({r['u']},{r['v']}) true Z {r['true_Z_m']}" if r
                                 else "no GT rows on this frame")
            self.draw_gt()
            self.fig.canvas.draw_idle()
            return
        step = {"right": 1, "left": -1, "pagedown": 10, "pageup": -10}.get(e.key)
        if step:
            self.k = (self.k + step) % len(self.pairs)
            self.load()
        elif e.key == "o":
            self.overlay = not self.overlay
            self.redraw()
        elif e.key == "c":
            self.fixed_scale = not self.fixed_scale
            self.redraw()
        elif e.key == "s":
            out = self.pairs[self.k][1].parent.parent / f"view_{self.pairs[self.k][0].stem}.png"
            self.fig.savefig(out, dpi=130)
            print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="output folder containing rgb/ and depth_mm/")
    ap.add_argument("--rgb-dir", help="folder of photos (overrides auto-detect)")
    ap.add_argument("--depth-dir", help="folder of depth maps (overrides auto-detect)")
    ap.add_argument("--depth-scale", type=float, default=None,
                    help="multiply raw depth values by this to get metres "
                         "(default: 0.001 for 16-bit PNG, 1 for float/npy)")
    ap.add_argument("--start", type=int, default=0, help="index of the first frame to show")
    ap.add_argument("--vmax", type=float, default=4.0, help="top of the fixed colour scale (m), key 'c'")
    ap.add_argument("--cmap", default="turbo", help="matplotlib colour map")
    ap.add_argument("--gt", nargs="?", const=str(DEFAULT_GT), default=None,
                    help=f"ground-truth mode; rows are appended to this CSV (default {DEFAULT_GT})")
    a = ap.parse_args()
    rgb, dep = pick_dirs(a)
    pairs = pair_frames(rgb, dep)
    print(f"{len(pairs)} frames  |  photos: {rgb}  |  depth: {dep}")
    for k in ("left", "right", "s", "c", "o", "u"):      # free these keys from matplotlib defaults
        for name, keys in plt.rcParams.items():
            if name.startswith("keymap.") and k in keys:
                keys.remove(k)
    Viewer(pairs, a)
    plt.show()


if __name__ == "__main__":
    main()
