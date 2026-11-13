"""Check the depth of any frame of a recorded run: pick it by frame number or by time.

Three panels, same colour scale:
  photo + lidar dots   the real lidar returns (coloured by their true distance)
  raw depth            Depth Anything output as-is (what plain setup-rgbd.sh gives)
  calibrated depth     after the per-frame lidar + floor correction

Hover over any panel: shows pixel, raw depth, calibrated depth, and the lidar's distance
if a lidar dot is within a few pixels (so you can compare against a real measurement).
The title line also gives how well this frame's calibrated depth agrees with its lidar.

Usage (any recorded run; depth must exist from ./setup-rgbd.sh <run>/frames first):
  uv run python -m multi_sg.check_frame                       # room_run1: shows frame count, asks
  uv run python -m multi_sg.check_frame 830                   # frame number
  uv run python -m multi_sg.check_frame 3:00                  # time since the start (m:ss)
  uv run python -m multi_sg.check_frame r                     # a random frame
  uv run python -m multi_sg.check_frame --run ~/rob535/data/new_room      # another run: asks
  uv run python -m multi_sg.check_frame 120 --run ~/rob535/data/new_room

The depth folder is found automatically as outputs/<run name>_<vitl|vitb|vits> (what
setup-rgbd.sh writes); override with --depth.

Controls:
  type in the "go to" box     a frame number, m:ss, 123s, or r (random), then Enter
  r                           jump to a random frame
  <- / ->                     previous / next frame
  down / up                   back / forward 1 second (~5 frames)
  Page Down / Page Up         back / forward 10 seconds
  left click                  pin a reading on all three panels (right click clears)
  c                           colour scale: fixed 0-5 m / auto per frame
  s                           save a PNG of the window to outputs/checks/
  q                           quit
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox

from multi_sg.data import DEFAULT_RUN, REPO, Run

DEFAULT_DEPTH = REPO / "outputs" / "room_run1_vitl"


def read_depth(p):
    if not p.exists():
        return None
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    d[d <= 0] = np.nan
    return d


class Checker:
    def __init__(self, run, raw_dir, cal_dir, start):
        self.run, self.raw_dir, self.cal_dir = run, raw_dir, cal_dir
        self.t0 = run.t[0]
        self.fps = (len(run) - 1) / (run.t[-1] - run.t[0])
        self.k = start
        self.auto, self.pins = False, []
        self.cal_info = {}
        pc = cal_dir.parent / "calib_params.csv"
        if pc.exists():
            self.cal_info = {int(r["frame"]): r for r in csv.DictReader(open(pc))}

        self.fig, self.axs = plt.subplots(1, 3, figsize=(17, 5.6))
        self.fig.subplots_adjust(left=0.01, right=0.95, top=0.83, bottom=0.11, wspace=0.03)
        for ax in self.axs:
            ax.set_xticks([]), ax.set_yticks([])
        self.im = [self.axs[0].imshow(np.zeros((480, 640, 3), np.uint8))]
        for ax in self.axs[1:]:
            self.im.append(ax.imshow(np.zeros((480, 640)), cmap="turbo", vmin=0, vmax=5))
            self.im[-1].cmap.set_bad("black")
        self.dots = self.axs[0].scatter([], [], c=[], s=10, cmap="turbo", vmin=0, vmax=5,
                                        edgecolors="black", linewidths=0.3)
        for ax, t in zip(self.axs, ("photo + lidar dots (true distance)", "raw network depth",
                                    "calibrated depth")):
            ax.set_title(t, fontsize=10)
        cax = self.fig.add_axes([0.955, 0.15, 0.01, 0.6])
        self.cbar = self.fig.colorbar(self.im[2], cax=cax)
        self.cbar.set_label("distance Z (m)")
        self.marks = [ax.plot([], [], "+", ms=16, mew=1.5, color="white")[0] for ax in self.axs]
        self.head = self.fig.text(0.01, 0.95, "", fontsize=11, weight="bold")
        self.sub = self.fig.text(0.01, 0.905, "", fontsize=9.5, color="0.3")
        self.readout = self.fig.text(0.01, 0.86, "", family="monospace", fontsize=10.5)
        bax = self.fig.add_axes([0.40, 0.025, 0.12, 0.05])
        self.box = TextBox(bax, f"go to (0-{int(run.idx[-1])}, m:ss, 123s or r)  ", initial="")
        self.box.on_submit(self.goto)
        self.msg = self.fig.text(0.54, 0.04, "", fontsize=10, color="firebrick")
        self.pin_art = []
        c = self.fig.canvas
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_press_event", self.on_click)
        c.mpl_connect("key_press_event", self.on_key)
        self.load()

    # ------------------------------------------------------------ helpers
    def parse(self, text):
        """frame number, m:ss(.s) or 12.5s -> run index (k), or None."""
        text = text.strip().lower()
        if text in ("r", "random"):
            return int(np.random.default_rng().integers(len(self.run)))
        m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", text)
        s = re.fullmatch(r"(\d+(?:\.\d+)?)s", text)
        if m or s:
            sec = int(m[1]) * 60 + float(m[2]) if m else float(s[1])
            return int(np.argmin(np.abs(self.run.t - self.t0 - sec)))
        if text.isdigit():
            i = int(text)
            return self.run._pos.get(i)
        return None

    def load(self):
        run = self.run
        self.k = int(np.clip(self.k, 0, len(run) - 1))
        i = int(run.idx[self.k])
        self.i = i
        f = run.frame(i)
        self.f = f
        self.raw = read_depth(self.raw_dir / f"{i:06d}.png")
        self.cal = read_depth(self.cal_dir / f"{i:06d}.png")
        u, v, Z, ins = f.project(f.scan_cam)
        self.lu, self.lv, self.lz = u[ins], v[ins], Z[ins]
        self.pins = []
        self.msg.set_text("" if self.raw is not None else f"no depth for frame {i} yet")
        self.redraw()

    def agreement(self):
        """Median |cal - lidar| / lidar over lidar dots on smooth depth (<6 m)."""
        if self.cal is None or len(self.lz) == 0:
            return ""
        ui = np.clip(np.round(self.lu).astype(int), 0, 639)
        vi = np.clip(np.round(self.lv).astype(int), 0, 479)
        out = []
        for name, d in (("calibrated", self.cal), ("raw", self.raw)):
            z = d[vi, ui]
            pad = np.pad(d, 4, mode="edge")
            win = np.stack([pad[vi + dy, ui + dx] for dy in range(9) for dx in range(9)], 1)
            smooth = (np.nanpercentile(win, 90, 1) - np.nanpercentile(win, 10, 1)) / np.nanmedian(win, 1) < 0.15
            m = smooth & np.isfinite(z) & (self.lz < 6)
            if m.sum() >= 5:
                rel = np.abs(z[m] - self.lz[m]) / self.lz[m]
                out.append(f"{name} {100 * np.median(rel):.0f}%")
        return ("agreement with this frame's lidar (median error, smooth points < 6 m): "
                + ", ".join(out)) if out else ""

    # ------------------------------------------------------------ drawing
    def redraw(self):
        f, i = self.f, self.i
        self.im[0].set_data(f.rgb)
        for im, d in zip(self.im[1:], (self.raw, self.cal)):
            im.set_data(d if d is not None else np.full((480, 640), np.nan))
        if self.auto and self.cal is not None:
            lo, hi = np.nanpercentile(self.cal, [2, 98])
        else:
            lo, hi = 0.0, 5.0
        for im in self.im[1:]:
            im.set_clim(lo, hi)
        self.dots.set_clim(lo, hi)
        self.dots.set_offsets(np.c_[self.lu, self.lv] if len(self.lu) else np.empty((0, 2)))
        self.dots.set_array(self.lz)
        t = self.run.t[self.k] - self.t0
        info = self.cal_info.get(i, {})
        meth = info.get("method", "?")
        if info.get("borrowed") == "1":
            meth += f" (borrowed from frame {info.get('fit_from')}: robot turning / no fit)"
        x, y, yaw = f.pose
        self.head.set_text(f"frame {i}  (of 0-{int(self.run.idx[-1])})   time {int(t // 60)}:{t % 60:04.1f}   "
                           f"pose x {x:+.2f} m, y {y:+.2f} m, heading {np.rad2deg(yaw):+.0f} deg"
                           f"{'   [TURNING]' if f.turning else ''}")
        self.sub.set_text(f"calibration: {meth}     {self.agreement()}")
        self.readout.set_text("hover over any panel to read depth")
        self.draw_pins()
        self.fig.canvas.draw_idle()

    def reading(self, u, v):
        def at(d):
            if d is None:
                return np.nan
            return float(np.nanmedian(d[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2]))
        raw, cal = at(self.raw), at(self.cal)
        lid = None
        if len(self.lu):
            dist = np.hypot(self.lu - u, self.lv - v)
            j = int(np.argmin(dist))
            if dist[j] < 6:
                lid = float(self.lz[j])
        return raw, cal, lid

    def draw_pins(self):
        for a in self.pin_art:
            a.remove()
        self.pin_art = []
        for u, v in self.pins:
            raw, cal, lid = self.reading(u, v)
            labels = (f"{cal:.2f} m" + (f" | lidar {lid:.2f}" if lid else ""), f"{raw:.2f} m", f"{cal:.2f} m")
            for ax, lab in zip(self.axs, labels):
                self.pin_art.append(ax.plot(u, v, "o", ms=6, mfc="none", mec="white", mew=1.5)[0])
                self.pin_art.append(ax.annotate(lab, (u, v), xytext=(6, -8), textcoords="offset points",
                                                color="white", fontsize=9,
                                                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65, lw=0)))

    # ------------------------------------------------------------ events
    def pixel(self, e):
        if e.inaxes not in self.axs or e.xdata is None:
            return None
        u, v = int(round(e.xdata)), int(round(e.ydata))
        return (u, v) if 0 <= u < 640 and 0 <= v < 480 else None

    def on_move(self, e):
        p = self.pixel(e)
        if p is None:
            return
        u, v = p
        raw, cal, lid = self.reading(u, v)
        for m in self.marks:
            m.set_data([u], [v])
        txt = f"pixel ({u:3d},{v:3d})   calibrated {cal:5.2f} m   raw {raw:5.2f} m"
        if lid is not None:
            txt += f"   lidar here {lid:5.2f} m  (calibrated error {100 * (cal - lid) / lid:+.0f}%)"
        self.readout.set_text(txt)
        self.fig.canvas.draw_idle()

    def on_click(self, e):
        p = self.pixel(e)
        if e.button == 3:
            self.pins = []
        elif e.button == 1 and p is not None:
            self.pins.append(p)
        else:
            return
        self.draw_pins()
        self.fig.canvas.draw_idle()

    def goto(self, text):
        if not text.strip():
            return
        k = self.parse(text)
        if k is None:
            self.msg.set_text(f"can't read {text!r}: use a frame number 0-{int(self.run.idx[-1])}, "
                              "m:ss, 123s or r")
            self.fig.canvas.draw_idle()
            return
        self.k = k
        self.box.set_val("")
        self.load()

    def on_key(self, e):
        if self.box.capturekeystrokes:          # typing in the box
            return
        step = {"right": 1, "left": -1, "up": round(self.fps), "down": -round(self.fps),
                "pageup": round(10 * self.fps), "pagedown": -round(10 * self.fps)}.get(e.key)
        if step:
            self.k += step
            self.load()
        elif e.key == "r":
            self.k = self.parse("r")
            self.load()
        elif e.key == "c":
            self.auto = not self.auto
            self.redraw()
        elif e.key == "s":
            out = REPO / "outputs" / "checks"
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"check_{self.i:06d}.png"
            self.fig.savefig(p, dpi=120)
            self.msg.set_text(f"saved {p}")
            print("saved", p)
            self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default=None,
                    help="frame number, m:ss, 123s or r (random). Omitted: ask in the terminal")
    ap.add_argument("--run", default=str(DEFAULT_RUN),
                    help="recorded run folder (contains frames/, frames.csv, scans/)")
    ap.add_argument("--depth", default=None,
                    help="folder with depth_mm/ and calibrated/depth_mm/ "
                         "(default: outputs/<run name>_<vitl|vitb|vits>)")
    a = ap.parse_args()
    run_dir = Path(a.run).expanduser()
    if run_dir.name == "frames":              # accept the frames folder too
        run_dir = run_dir.parent
    if not (run_dir / "frames.csv").is_file():
        raise SystemExit(f"{run_dir} is not a recorded run (needs frames/, frames.csv, scans/)")
    run = Run(run_dir)
    if a.depth:
        base = Path(a.depth).expanduser()
    else:
        cands = [REPO / "outputs" / f"{run_dir.name}_{e}" for e in ("vitl", "vitb", "vits")]
        base = next((c for c in cands if (c / "calibrated" / "depth_mm").is_dir()), None)
        if base is None:
            raise SystemExit(f"no calibrated depth for {run_dir.name}: run first\n"
                             f"  ./setup-rgbd.sh {run_dir / 'frames'}")
    n_cal = len(list((base / "calibrated" / "depth_mm").glob("*.png")))
    dur = run.t[-1] - run.t[0]
    last = int(run.idx[-1])
    print(f"run {run_dir.name}: {len(run)} frames (numbers 0-{last}), {int(dur // 60)}:{dur % 60:04.1f} long, "
          f"~{(len(run) - 1) / dur:.1f} frames/s")
    print(f"depth: {base}  ({n_cal} of {len(run)} frames calibrated)")
    for k in ("left", "right", "up", "down", "pageup", "pagedown", "c", "s"):
        for name, keys in plt.rcParams.items():
            if name.startswith("keymap.") and k in keys:
                keys.remove(k)
    ch = Checker.__new__(Checker)        # parse() needs run/t0 before __init__ finishes
    ch.run, ch.t0 = run, run.t[0]
    which = a.which
    while True:
        if which is None:
            which = input(f"which frame? (0-{last}, m:ss, 123s, r = random): ").strip() or "r"
        k = Checker.parse(ch, which)
        if k is not None:
            break
        print(f"  can't read {which!r}: use a frame number 0-{last}, m:ss, 123s or r")
        which = None
    Checker(run, base / "depth_mm", base / "calibrated" / "depth_mm", k)
    plt.show()


if __name__ == "__main__":
    main()
