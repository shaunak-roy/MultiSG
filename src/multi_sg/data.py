"""Loader for a recorded MBot run (record_room.py output), e.g. ~/rob535/data/room_run1.

    run/frames/NNNNNN.jpg   640x480 photos (already rotated 180 deg by camera_ros -> no flip here)
    run/frames.csv          i, t, file, x, y, yaw, scan_dt   (pose = nearest /odom to the photo)
    run/scans/NNNNNN.npz    ranges, angles (rad), t          (nearest /scan to the photo)

Frames (all 4x4 homogeneous transforms, "T_a_b" maps points in b to points in a):
    world  : odom frame, z up
    robot  : base frame, x forward, y left, z up, origin on the floor
    cam    : OpenCV optical frame, X right, Y down, Z forward
    lidar  : scan frame, x along angle 0, z up

Usage:
    from multi_sg.data import Run
    run = Run("~/rob535/data/room_run1")
    f = run.frame(120)
    f.rgb, f.K, f.T_world_cam, f.scan_cam, f.turning
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MOUNT = REPO / "config" / "mbot_mount.yaml"
DEFAULT_RUN = Path("~/rob535/data/room_run1").expanduser()

# frames whose scan was taken while turning faster than this (yaw change over the
# photo-scan time gap) are flagged; their lidar shouldn't be used for calibration
TURN_THRESH_DEG = 1.0


# ---------------------------------------------------------------- transforms
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def pose2d_T(x, y, yaw):
    """T_world_robot from a planar odom pose."""
    return make_T(rot_z(yaw), [x, y, 0.0])


def transform(T, pts):
    """Apply 4x4 T to (N,3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- mounting
# robot axes expressed as optical axes: optical X = -robot y, Y = -robot z, Z = robot x
R_ROBOT_OPTICAL = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])


@dataclass
class Mount:
    T_robot_cam: np.ndarray
    T_robot_lidar: np.ndarray
    lidar_flip: bool
    K: np.ndarray
    raw: dict

    @classmethod
    def load(cls, path=DEFAULT_MOUNT):
        cfg = yaml.safe_load(Path(path).read_text())
        c, l, k = cfg["camera"], cfg["lidar"], cfg["intrinsics"]
        # yaw about z, then pitch about the (yawed) left axis: +pitch tips forward down
        R_cam = rot_z(np.deg2rad(c["yaw_deg"])) @ rot_y(np.deg2rad(c["pitch_deg"])) @ R_ROBOT_OPTICAL
        T_rc = make_T(R_cam, [c["x"], c["y"], c["z"]])
        T_rl = make_T(rot_z(np.deg2rad(l["yaw_deg"])), [l["x"], l["y"], l["z"]])
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        return cls(T_rc, T_rl, bool(l.get("flip", False)), K, cfg)


# ---------------------------------------------------------------- frames
@dataclass
class Frame:
    i: int
    t: float                 # photo time (s)
    rgb: np.ndarray          # HxWx3 uint8, RGB order
    K: np.ndarray            # 3x3
    pose: tuple              # (x, y, yaw) odom at photo time
    T_world_robot: np.ndarray
    T_world_cam: np.ndarray
    scan_t: float
    scan_dt: float           # scan time - photo time (s, signed)
    scan_lidar: np.ndarray   # (N,3) valid returns in the lidar frame
    scan_robot_now: np.ndarray  # (N,3) same points in the robot frame AT PHOTO TIME
    yaw_rate: float          # rad/s around the photo time
    turning: bool            # |yaw change over the scan gap| > TURN_THRESH_DEG

    @property
    def scan_cam(self):
        """Lidar points in the camera optical frame (motion-compensated to photo time)."""
        return transform(np.linalg.inv(self.T_world_cam) @ self.T_world_robot, self.scan_robot_now)

    def project(self, pts_cam, zmin=0.05):
        """(N,3) camera points -> (u, v, Z, mask_in_image). No lens distortion model."""
        Z = pts_cam[:, 2]
        ok = Z > zmin
        uv = np.full((len(Z), 2), np.nan)
        uv[ok] = (pts_cam[ok] @ self.K.T)[:, :2] / Z[ok, None]
        H, W = self.rgb.shape[:2]
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv[:, 0], uv[:, 1], Z, inside


class Run:
    def __init__(self, root=DEFAULT_RUN, mount=DEFAULT_MOUNT, rmin=0.05, rmax=12.0):
        self.root = Path(root).expanduser()
        self.mount = mount if isinstance(mount, Mount) else Mount.load(mount)
        self.rmin, self.rmax = rmin, rmax
        with open(self.root / "frames.csv") as fh:
            rows = list(csv.DictReader(fh))
        self.idx = np.array([int(r["i"]) for r in rows])
        self.files = [r["file"] for r in rows]
        self.t = np.array([float(r["t"]) for r in rows])
        self.xyyaw = np.array([[float(r["x"]), float(r["y"]), float(r["yaw"])] for r in rows])
        self._pos = {i: k for k, i in enumerate(self.idx)}
        # yaw rate from neighbouring samples (central difference, unwrapped)
        yaw_u = np.unwrap(self.xyyaw[:, 2])
        self.yaw_rate = np.gradient(yaw_u, self.t) if len(self.t) > 1 else np.zeros(1)

    def __len__(self):
        return len(self.idx)

    def pose_at(self, t):
        """Odom pose interpolated at time t (linear in x, y, unwrapped yaw)."""
        yaw_u = np.unwrap(self.xyyaw[:, 2])
        x = np.interp(t, self.t, self.xyyaw[:, 0])
        y = np.interp(t, self.t, self.xyyaw[:, 1])
        return x, y, wrap(np.interp(t, self.t, yaw_u))

    def scan(self, i):
        d = np.load(self.root / "scans" / f"{i:06d}.npz")
        r, a = d["ranges"].astype(np.float64), d["angles"].astype(np.float64)
        ok = np.isfinite(r) & (r > self.rmin) & (r < self.rmax)
        r, a = r[ok], a[ok]
        if self.mount.lidar_flip:
            a = -a
        pts = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], 1)
        return float(d["t"]), pts

    def frame(self, i, load_rgb=True):
        k = self._pos[i]
        x, y, yaw = self.xyyaw[k]
        T_wr = pose2d_T(x, y, yaw)
        m = self.mount
        rgb = None
        if load_rgb:
            bgr = cv2.imread(str(self.root / "frames" / self.files[k]), cv2.IMREAD_COLOR)
            rgb = bgr[..., ::-1].copy()
        scan_t, pts_l = self.scan(i)
        # scan taken at scan_t: put it in the world with the pose at scan_t, then back into
        # the robot frame at photo time (corrects the 0.1-0.25 s photo/scan gap while moving)
        T_wr_scan = pose2d_T(*self.pose_at(scan_t))
        pts_world = transform(T_wr_scan @ m.T_robot_lidar, pts_l)
        pts_robot_now = transform(np.linalg.inv(T_wr), pts_world)
        dt = scan_t - self.t[k]
        rate = float(self.yaw_rate[k])
        return Frame(
            i=int(i), t=float(self.t[k]), rgb=rgb, K=m.K, pose=(x, y, yaw),
            T_world_robot=T_wr, T_world_cam=T_wr @ m.T_robot_cam,
            scan_t=scan_t, scan_dt=dt, scan_lidar=pts_l, scan_robot_now=pts_robot_now,
            yaw_rate=rate, turning=abs(rate * dt) > np.deg2rad(TURN_THRESH_DEG),
        )

    def turning_mask(self):
        """Per-frame turning flag without loading photos (uses frames.csv scan_dt)."""
        with open(self.root / "frames.csv") as fh:
            sdt = np.array([float(r["scan_dt"]) for r in csv.DictReader(fh)])
        return np.abs(self.yaw_rate * sdt) > np.deg2rad(TURN_THRESH_DEG)


if __name__ == "__main__":
    import sys
    run = Run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN)
    tm = run.turning_mask()
    path = np.sum(np.linalg.norm(np.diff(run.xyyaw[:, :2], axis=0), axis=1))
    print(f"{len(run)} frames, {run.t[-1] - run.t[0]:.0f} s, odom path {path:.1f} m, "
          f"{tm.sum()} turning ({100 * tm.mean():.0f}%)")
    f = run.frame(run.idx[0])
    u, v, Z, ins = f.project(f.scan_cam)
    print(f"frame {f.i}: {len(f.scan_lidar)} lidar returns, {ins.sum()} land in the image, "
          f"scan_dt {f.scan_dt:+.3f} s")
