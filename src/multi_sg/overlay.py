"""Draw lidar returns on the photos to check the camera/lidar mounting (plan step 4).

Each output image has two panels:
  left   the photo with every lidar return that projects into it, coloured by depth Z
         (turbo: red = near, blue = far; scale fixed 0..--zmax m so frames compare).
         With a correct mounting the dots sit on whatever is ~lidar height above the floor:
         box tops, chair bases, cabinet sides, wall skirting.
  right  top-down view of the scan in the robot frame (x forward = up), with the camera's
         field of view as a wedge. Dots inside the wedge are the ones drawn on the photo.
         If the obvious structure in the photo appears outside the wedge, the lidar yaw /
         flip is wrong.

Usage:
  uv run python -m multi_sg.overlay                         # 5 frames spread over the run
  uv run python -m multi_sg.overlay --frames 0 400 900
  uv run python -m multi_sg.overlay --n 12 --out outputs/overlays_try2
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from multi_sg.data import DEFAULT_MOUNT, DEFAULT_RUN, REPO, Run

BEV_PX, BEV_M = 480, 4.0      # top-down panel: 480 px covers +-4 m


def colour(z, zmax):
    t = np.clip(255 * (1 - z / zmax), 0, 255).astype(np.uint8)   # near -> red end
    return cv2.applyColorMap(t.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def pick_frames(run, n):
    """n frames evenly over the run, each moved to the nearest non-turning frame."""
    turning = run.turning_mask()
    good = np.flatnonzero(~turning)
    targets = np.linspace(0, len(run) - 1, n).round().astype(int)
    return [int(run.idx[good[np.argmin(np.abs(good - t))]]) for t in targets]


def photo_panel(f, zmax):
    img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
    u, v, Z, ins = f.project(f.scan_cam)
    order = np.argsort(-Z[ins])                     # far first so near dots draw on top
    uu, vv, zz = u[ins][order], v[ins][order], Z[ins][order]
    for (x, y), c in zip(np.stack([uu, vv], 1).round().astype(int), colour(zz, zmax)):
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 2, tuple(int(k) for k in c), -1, cv2.LINE_AA)
    # horizontal line where the lidar plane would appear at infinity (sanity: the dots
    # should converge toward it with distance)
    return img, int(ins.sum()), zz


def bev_panel(f, mount, zmax):
    S = BEV_PX / (2 * BEV_M)
    img = np.full((BEV_PX, BEV_PX, 3), 255, np.uint8)
    c0 = (BEV_PX // 2, BEV_PX // 2)

    def to_px(x, y):                                # robot x forward = up, y left = left
        return (np.round(c0[0] - y * S).astype(int), np.round(c0[1] - x * S).astype(int))

    for r in (1, 2, 3, 4):
        cv2.circle(img, c0, int(r * S), (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r}m", (c0[0] + 3, c0[1] - int(r * S) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    # camera FOV wedge
    T_rc = mount.T_robot_cam
    W, K = f.rgb.shape[1], f.K
    cam_o = T_rc[:3, 3]
    for u in (0, W - 1):
        ray = T_rc[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], 0, 1.0])
        end = cam_o + ray / np.linalg.norm(ray[:2]) * BEV_M * 1.5
        cv2.line(img, tuple(int(a) for a in to_px(*cam_o[:2])), tuple(int(a) for a in to_px(*end[:2])),
                 (90, 170, 90), 1, cv2.LINE_AA)
    p = f.scan_robot_now
    pc = f.scan_cam
    _, _, Z, ins = f.project(pc)
    xs, ys = to_px(p[:, 0], p[:, 1])
    for x, y in zip(xs[~ins], ys[~ins]):
        cv2.circle(img, (int(x), int(y)), 1, (150, 150, 150), -1)
    for (x, y), c in zip(zip(xs[ins], ys[ins]), colour(Z[ins], zmax)):
        cv2.circle(img, (int(x), int(y)), 2, tuple(int(k) for k in c), -1)
    # robot + lidar angle-0 direction
    cv2.circle(img, c0, 6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, c0, (c0[0], c0[1] - 25), (0, 0, 0), 1, cv2.LINE_AA, tipLength=0.3)
    T_rl = mount.T_robot_lidar
    a0 = transform_dir(T_rl, np.array([0.4, 0, 0]))
    cv2.arrowedLine(img, c0, tuple(int(a) for a in to_px(*a0[:2])), (200, 60, 200), 1,
                    cv2.LINE_AA, tipLength=0.2)
    cv2.putText(img, "black: robot fwd  magenta: lidar angle 0", (6, BEV_PX - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def transform_dir(T, d):
    return T[:3, :3] @ d


def render(run, i, zmax):
    f = run.frame(i)
    left, n_in, zz = photo_panel(f, zmax)
    right = bev_panel(f, run.mount, zmax)
    canvas = np.full((480 + 60, 640 + BEV_PX, 3), 255, np.uint8)
    canvas[60:, :640], canvas[60:, 640:] = left, right
    c, l = run.mount.raw["camera"], run.mount.raw["lidar"]
    zs = f"Z {zz.min():.2f}-{zz.max():.2f} m" if n_in else "no dots in image"
    lines = [
        f"frame {f.i}   pose ({f.pose[0]:+.2f}, {f.pose[1]:+.2f}, {np.rad2deg(f.pose[2]):+.0f} deg)"
        f"   scan_dt {f.scan_dt:+.3f} s   yaw rate {np.rad2deg(f.yaw_rate):+.0f} deg/s"
        f"{'   TURNING' if f.turning else ''}   {n_in} dots, {zs}",
        f"cam x{c['x']:.3f} y{c['y']:.3f} z{c['z']:.3f} pitch{c['pitch_deg']:+.1f} yaw{c['yaw_deg']:+.1f}"
        f"   |   lidar x{l['x']:.3f} y{l['y']:.3f} z{l['z']:.3f} yaw{l['yaw_deg']:+.1f}"
        f" flip={l.get('flip', False)}   |   colour 0 (red) - {zmax:g} m (blue)",
    ]
    for k, s in enumerate(lines):
        cv2.putText(canvas, s, (8, 22 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--mount", default=str(DEFAULT_MOUNT))
    ap.add_argument("--frames", type=int, nargs="*", help="frame indices (default: --n spread out)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--zmax", type=float, default=4.0)
    ap.add_argument("--out", default=str(REPO / "outputs" / "overlays"))
    a = ap.parse_args()
    run = Run(a.run, a.mount)
    frames = a.frames or pick_frames(run, a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in frames:
        p = out / f"overlay_{i:06d}.jpg"
        cv2.imwrite(str(p), render(run, i, a.zmax), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(p)


if __name__ == "__main__":
    main()
