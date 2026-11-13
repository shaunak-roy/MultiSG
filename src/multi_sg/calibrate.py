"""Per-frame metric correction of network depth with lidar + floor anchors (plan step 5).

Network depth from Depth Anything V2 has good ordering but a wrong, depth-dependent scale
(near objects ~2x too far from our 7 cm-high camera). Per frame we fit, in inverse depth,

        1/Z_true = a * (1/Z_net) + b

using two anchor sets whose true Z we know without a tape:
  lidar  each scan return projected into the photo (motion-compensated, 18 cm plane);
         true Z = the point's camera-frame Z. Image columns where the net depth is at an
         edge (big 5x5 spread) are dropped.
  floor  pixels below the horizon whose floor-ray is shorter than the lidar range in that
         column (so the floor is visible there); true Z from geometry,
         Z = h / tan(pitch + atan((v - cy)/fy)), with h, pitch from config/mbot_mount.yaml.

Fit: RANSAC on 2-point models, inliers within 8% depth error, then weighted least squares
on inliers. Anchors are split 70/30 per frame; methods are scored on the held-out 30% of
lidar points (and held-out floor points), never on what they were fitted to.

Methods compared:
  raw        network output as is (= what setup-rgbd.sh produces)
  median     one scale per frame, Z = s * Z_net, s = median(Z_true / Z_net) over lidar
  inv_lidar  inverse-depth affine fitted on lidar only
  inv_both   inverse-depth affine fitted on lidar + floor

  uv run python -m multi_sg.calibrate --depth outputs/room_run1_vits/depth_mm --every 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from multi_sg.data import DEFAULT_MOUNT, DEFAULT_RUN, REPO, Run

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ anchors
def load_depth(path):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    d[d <= 0] = np.nan
    return d


def lidar_anchors(f, dnet, zmax=6.0, edge_rel=0.15, win=9, jump=0.08):
    """(u, v, Z_true, Z_net) for lidar returns that land in the image on smooth depth.

    Dropped: returns next to a range jump in the scan (beam slipping past a chair leg,
    edge of an object) and pixels where the network depth varies a lot in a win x win
    window (the network blurs thin / see-through things, so pixel and beam disagree)."""
    u, v, Z, ins = f.project(f.scan_cam)
    r = np.linalg.norm(f.scan_lidar[:, :2], axis=1)          # scan order = array order
    rel_prev = np.abs(np.diff(r, prepend=r[0])) / r
    rel_next = np.abs(np.diff(r, append=r[-1])) / r
    smooth = (rel_prev < jump) & (rel_next < jump)
    ok = ins & (Z < zmax) & smooth
    u, v, Z = u[ok], v[ok], Z[ok]
    H, W = dnet.shape
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    # window: median for the value, spread for edge rejection
    hw = win // 2
    pad = np.pad(dnet, hw, mode="edge")
    W_ = np.stack([pad[vi + dy, ui + dx] for dy in range(win) for dx in range(win)], 1)
    med = np.nanmedian(W_, 1)
    spread = (np.nanpercentile(W_, 90, 1) - np.nanpercentile(W_, 10, 1)) / med
    keep = np.isfinite(med) & (spread < edge_rel)
    return u[keep], v[keep], Z[keep], med[keep]


def floor_Z(v, K, h, pitch_down):
    """True Z (camera axis) of a floor pixel in row v, for a camera at height h tilted down."""
    ang = pitch_down + np.arctan((v - K[1, 2]) / K[1, 1])   # angle of the ray below horizontal
    with np.errstate(divide="ignore", invalid="ignore"):
        rng_h = np.where(ang > np.deg2rad(1.5), h / np.tan(ang), np.nan)   # horizontal distance
    # convert horizontal distance to optical-axis Z: Z = horiz * cos(pitch) + ... small-angle exact form
    return rng_h * np.cos(np.arctan((v - K[1, 2]) / K[1, 1])) / np.cos(ang)


def floor_anchors(f, dnet, h, pitch_down, step=8, max_dist=3.0, margin=0.15):
    """Floor pixels on a grid, kept only where the floor is visible (closer than the
    lidar obstacle in that image column, with a margin)."""
    H, W = dnet.shape
    K = f.K
    vs = np.arange(H - 1, 0, -step)
    us = np.arange(step // 2, W, step)
    Zf = floor_Z(vs.astype(float), K, h, pitch_down)
    # obstacle distance per image column from lidar (Z of nearest return in that column band)
    lu, lv, lZ, ins = f.project(f.scan_cam)
    col_obst = np.full(W, np.inf)
    if ins.any():
        for uu, zz in zip(np.clip(np.round(lu[ins]).astype(int), 0, W - 1), lZ[ins]):
            lo, hi = max(uu - 6, 0), min(uu + 7, W)
            col_obst[lo:hi] = np.minimum(col_obst[lo:hi], zz)
    out = []
    for v, z in zip(vs, Zf):
        if not np.isfinite(z) or z > max_dist:
            continue
        for u in us:
            if z < col_obst[u] - margin:
                dn = dnet[v, u]
                if np.isfinite(dn):
                    out.append((u, v, z, dn))
    if not out:
        return (np.empty(0),) * 4
    a = np.array(out, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# ------------------------------------------------------------------ fitting
def fit_inv_affine(zt, zn, w=None, iters=200, tol=0.08):
    """RANSAC + weighted LSQ for 1/zt = a/zn + b. Returns (a, b, inlier_mask) or None."""
    n = len(zt)
    if n < 6:
        return None
    x, y = 1.0 / zn, 1.0 / zt
    w = np.ones(n) if w is None else w
    best, best_score = None, -1
    for _ in range(iters):
        i, j = RNG.choice(n, 2, replace=False)
        if abs(x[i] - x[j]) < 1e-3:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        b = y[i] - a * x[i]
        if a <= 0:
            continue
        zp = 1.0 / np.maximum(a * x + b, 1e-3)
        inl = np.abs(zp - zt) / zt < tol
        score = w[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 4:
        return None
    for _ in range(2):   # refit on inliers, re-select
        A = np.stack([x[best], np.ones(best.sum())], 1) * np.sqrt(w[best])[:, None]
        a, b = np.linalg.lstsq(A, y[best] * np.sqrt(w[best]), rcond=None)[0]
        if a <= 0:
            return None
        zp = 1.0 / np.maximum(a * x + b, 1e-3)
        new = np.abs(zp - zt) / zt < tol
        if new.sum() < 4:
            break
        best = new
    return float(a), float(b), best


# Each model is linear in its parameters theta after a transform, with residual ~ relative
# depth error:  inv     zt*(a/zn + b) - 1 = 0      (Z = 1/(a/zn + b))
#               loglin  log zt = c0 + c1 log zn     (power law Z = k * zn^c1)
#               logquad log zt = c0 + c1 L + c2 L^2 (L = log zn)
def _hinge(L, knots):
    return [np.maximum(0.0, L - k) for k in knots]


def _design(kind, zt, zn, knots=()):
    if kind == "logpwl":
        L = np.log(zn)
        return np.stack([np.ones_like(L), L] + _hinge(L, knots), 1), np.log(zt)
    if kind == "inv":
        return np.stack([zt / zn, zt], 1), np.ones_like(zt)
    L = np.log(zn)
    cols = [np.ones_like(L), L] + ([L * L] if kind == "logquad" else [])
    return np.stack(cols, 1), np.log(zt)


def predict(kind, th, zn):
    if kind == "logpwl":                      # th = (coeffs..., knots...)
        nk = (len(th) - 2) // 2
        c, knots = th[:2 + nk], th[2 + nk:]
        zn = np.asarray(zn, float)
        L = np.log(zn.ravel())
        return np.exp(np.stack([np.ones_like(L), L] + _hinge(L, knots), 1) @ c).reshape(zn.shape)
    if kind == "inv":
        return apply_inv(th[0], th[1], zn)
    L = np.log(zn)
    y = th[0] + th[1] * L + (th[2] * L * L if kind == "logquad" else 0)
    return np.exp(y)


def _valid(kind, th, zn):
    """Reject non-monotone fits over the observed range (more net depth must mean more depth)."""
    if kind == "inv":
        return th[0] > 0
    lo, hi = np.log(np.nanmin(zn)), np.log(np.nanmax(zn))
    d = lambda L: th[1] + (2 * th[2] * L if kind == "logquad" else 0)
    return d(lo) > 0.05 and d(hi) > 0.05


def fit_pwl(zt, zn, w, nknots=3, tol=0.08):
    """Monotone piecewise-linear log Z_true vs log Z_net. Inliers come from a logquad
    RANSAC; knots at quantiles of the inliers' log net depth; slopes kept >= 0.05."""
    th0 = fit_model("logquad", zt, zn, w, tol=tol)
    if th0 is None:
        return None
    inl = np.abs(predict("logquad", th0, zn) - zt) / zt < tol * 1.5
    if inl.sum() < 12:
        return None
    L = np.log(zn[inl])
    knots = np.quantile(L, np.linspace(0, 1, nknots + 2)[1:-1])
    A, y = _design("logpwl", zt[inl], zn[inl], knots)
    ww = w[inl].copy()
    for _ in range(6):
        sw = np.sqrt(ww)
        c = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(A @ c - y)
        ww = w[inl] * np.minimum(1.0, 0.03 / np.maximum(r, 1e-9))
    slopes = np.cumsum(c[1:])
    if (slopes < 0.05).any():
        return th0 and np.concatenate([[np.nan]])  # signal: non-monotone -> caller falls back
    return np.concatenate([c, knots])


def fit_model(kind, zt, zn, w=None, iters=300, tol=0.08):
    """RANSAC (minimal samples, relative-error inliers) + weighted Huber IRLS on inliers."""
    n, k = len(zt), (3 if kind == "logquad" else 2)
    if n < 3 * k:
        return None
    w = np.ones(n) if w is None else w
    A, y = _design(kind, zt, zn)
    best, best_score = None, -1.0
    for _ in range(iters):
        idx = RNG.choice(n, k, replace=False)
        try:
            th = np.linalg.solve(A[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        if not _valid(kind, th, zn):
            continue
        inl = np.abs(predict(kind, th, zn) - zt) / zt < tol
        sc = w[inl].sum()
        if sc > best_score:
            best, best_score = inl, sc
    if best is None or best.sum() < 2 * k:
        return None
    th = None
    for _ in range(3):
        ww = w[best].copy()
        for _ in range(5):                                   # Huber IRLS
            sw = np.sqrt(ww)
            th = np.linalg.lstsq(A[best] * sw[:, None], y[best] * sw, rcond=None)[0]
            r = np.abs(A[best] @ th - y[best])
            ww = w[best] * np.minimum(1.0, 0.03 / np.maximum(r, 1e-9))
        if not _valid(kind, th, zn):
            return None
        new = np.abs(predict(kind, th, zn) - zt) / zt < tol
        if new.sum() < 2 * k:
            break
        best = new
    return th


def apply_inv(a, b, zn):
    inv = a / zn + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(inv > 1e-3, 1.0 / inv, np.nan)


# ------------------------------------------------------------------ main loop
def split(n, frac=0.3):
    m = np.zeros(n, bool)
    m[RNG.choice(n, int(round(frac * n)), replace=False)] = True
    return m      # True = held out


def calibrate_frame(f, dnet, h, pitch_down):
    lu, lv, lz, ln = lidar_anchors(f, dnet)
    fu, fv, fz, fn = floor_anchors(f, dnet, h, pitch_down)
    if len(lz) < 10:
        return None
    lt, ft = split(len(lz)), split(len(fz))
    res = dict(i=f.i, n_lidar=len(lz), n_floor=len(fz), turning=f.turning)
    preds = {}
    # raw
    preds["raw"] = (lambda zn: zn)
    # median scale on lidar train
    s = float(np.median(lz[~lt] / ln[~lt]))
    preds["median"] = (lambda zn, s=s: s * zn)
    res["scale"] = s
    # old inverse affine (inverse-depth residuals), kept for comparison
    fl = fit_inv_affine(lz[~lt], ln[~lt])
    if fl:
        preds["inv_lidar"] = (lambda zn, p=fl: apply_inv(p[0], p[1], zn))
    zt = np.concatenate([lz[~lt], fz[~ft]])
    zn = np.concatenate([ln[~lt], fn[~ft]])
    # balance the two sets so the many floor pixels don't swamp the lidar
    w = np.concatenate([np.full((~lt).sum(), 1.0 / max((~lt).sum(), 1)),
                        np.full((~ft).sum(), 1.0 / max((~ft).sum(), 1))])
    # floor anchors weighted by geometric reliability: a tilt error dtheta gives a relative
    # depth error ~ dtheta / theta (theta = ray angle below horizontal), so weight ~ theta^2
    if len(fz) >= 6:
        theta = np.arctan(f.mount_h / fz[~ft]) if hasattr(f, "mount_h") else None
    if len(fz) >= 6 and theta is not None:
        rel_sig = np.deg2rad(0.5) / theta                      # 0.5 deg tilt uncertainty
        wf = 1.0 / (rel_sig ** 2 + 0.03 ** 2)                  # + 3% network noise floor
        wf = wf / wf.sum()
        wl = np.full((~lt).sum(), 1.0 / max((~lt).sum(), 1))
        w_rel = np.concatenate([wl, wf])
        th = fit_model("logquad", zt, zn, w_rel)
        if th is not None:
            preds["logquad_fw"] = (lambda zn, t=th: predict("logquad", t, zn))
        thp = fit_pwl(zt, zn, w_rel)
        if thp is not None and np.isfinite(thp).all():
            preds["logpwl_fw"] = (lambda zn, t=thp: predict("logpwl", t, zn))
        elif th is not None:
            preds["logpwl_fw"] = preds["logquad_fw"]
    if len(fz) >= 6:
        fb = fit_inv_affine(zt, zn, w)
        if fb:
            preds["inv_both"] = (lambda zn, p=fb: apply_inv(p[0], p[1], zn))
        for kind in ("inv", "loglin", "logquad"):
            th = fit_model(kind, zt, zn, w)
            if th is not None:
                preds[f"{kind}_rel"] = (lambda zn, k=kind, t=th: predict(k, t, zn))
                res[f"{kind}_rel_theta"] = " ".join(f"{x:.4f}" for x in th)
    # score on held-out anchors
    for name, fn_ in preds.items():
        for tag, zt, zn, held in (("lidar", lz, ln, lt), ("floor", fz, fn, ft)):
            if held.sum() == 0:
                continue
            zp = fn_(zn[held])
            rel = np.abs(zp - zt[held]) / zt[held]
            res[f"{name}_{tag}_absrel"] = float(np.nanmedian(rel))
            res[f"{name}_{tag}_n"] = int(held.sum())
            # near-range lidar (< 1.5 m) separately: that's where raw was worst
            if tag == "lidar":
                near = zt[held] < 1.5
                if near.sum() >= 3:
                    res[f"{name}_near_absrel"] = float(np.nanmedian(rel[near]))
    return res, preds, (lu, lv, lz, ln), (fu, fv, fz, fn)


METHODS = ["raw", "median", "inv_lidar", "inv_both", "inv_rel", "loglin_rel", "logquad_rel",
           "logquad_fw", "logpwl_fw"]


def summarize(rows):
    out = {}
    for m in METHODS:
        out[m] = {}
        for k in ("lidar", "floor", "near"):
            v = np.array([r.get(f"{m}_{k}_absrel", np.nan) for r in rows], float)
            v = v[np.isfinite(v)]
            if v.size:
                out[m][k] = dict(median=float(np.median(v) * 100), mean=float(np.mean(v) * 100),
                                 p90=float(np.percentile(v, 90) * 100), frames=int(v.size))
    return out


def print_table(summ, title):
    print(f"\n{title}\n  held-out median |Z - Z_true| / Z_true per frame, then median over frames (%)")
    print(f"  {'method':<12} {'lidar (all)':>10} {'lidar <1.5m':>12} {'floor':>8}   frames")
    for m in METHODS:
        s = summ.get(m, {})
        g = lambda k: f"{s[k]['median']:.1f}" if k in s else "  -"
        print(f"  {m:<12} {g('lidar'):>10} {g('near'):>12} {g('floor'):>8}   {s.get('lidar', {}).get('frames', 0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--mount", default=str(DEFAULT_MOUNT))
    ap.add_argument("--depth", default=str(REPO / "outputs" / "room_run1_vits" / "depth_mm"))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--pitch", type=float, default=None, help="override camera pitch (deg, + = down)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="write calibrated depth for EVERY frame to <depth>/../calibrated/")
    a = ap.parse_args()
    run = Run(a.run, a.mount)
    cam = run.mount.raw["camera"]
    h = cam["z"]
    pitch = np.deg2rad(cam["pitch_deg"] if a.pitch is None else a.pitch)
    if a.pitch is not None:   # rebuild the camera transform with the overridden pitch
        import copy
        from multi_sg.data import Mount
        raw = copy.deepcopy(run.mount.raw)
        raw["camera"]["pitch_deg"] = a.pitch
        tmp = Path(REPO / "outputs" / ".mount_override.yaml")
        import yaml
        tmp.write_text(yaml.safe_dump(raw))
        run = Run(a.run, Mount.load(tmp))
    dep = Path(a.depth)
    if a.apply:
        out = Path(a.out) if a.out else dep.parent / "calibrated"
        apply_all(run, dep, out, h, pitch)
        return None
    rows = []
    turning = run.turning_mask()
    for k in range(0, len(run), a.every):
        if turning[k]:
            continue
        i = int(run.idx[k])
        p = dep / f"{i:06d}.png"
        if not p.exists():
            continue
        f = run.frame(i, load_rgb=False)
        f.rgb = np.zeros((480, 640, 3), np.uint8)   # project() only needs the shape
        f.mount_h = h
        r = calibrate_frame(f, load_depth(p), h, pitch)
        if r:
            rows.append(r[0])
    summ = summarize(rows)
    out = Path(a.out) if a.out else dep.parent / "calib"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(dict(pitch_deg=float(np.rad2deg(pitch)), summary=summ), indent=2))
    if not a.quiet:
        print_table(summ, f"{dep.parent.name}  pitch {np.rad2deg(pitch):+.1f} deg  ({len(rows)} frames)")
    return summ



# ------------------------------------------------------------------ apply to every frame
def fit_full(f, dnet, h, pitch_down):
    """Final per-frame model on ALL anchors (no hold-out). Fallback chain:
    logpwl_fw -> logquad_fw -> inv (lidar only) -> median scale. Returns (name, fn) or None."""
    lu, lv, lz, ln = lidar_anchors(f, dnet)
    fu, fv, fz, fn = floor_anchors(f, dnet, h, pitch_down)
    if len(lz) < 10:
        return None
    if len(fz) >= 6:
        theta = np.arctan(h / fz)
        wf = 1.0 / ((np.deg2rad(0.5) / theta) ** 2 + 0.03 ** 2)
        w = np.concatenate([np.full(len(lz), 1.0 / len(lz)), wf / wf.sum()])
        zt, zn = np.concatenate([lz, fz]), np.concatenate([ln, fn])
        th = fit_pwl(zt, zn, w)
        if th is not None and np.isfinite(th).all():
            return "logpwl_fw", (lambda d, t=th: predict("logpwl", t, d))
        th = fit_model("logquad", zt, zn, w)
        if th is not None:
            return "logquad_fw", (lambda d, t=th: predict("logquad", t, d))
    fl = fit_inv_affine(lz, ln)
    if fl:
        return "inv_lidar", (lambda d, p=fl: apply_inv(p[0], p[1], d))
    s = float(np.median(lz / ln))
    return "median", (lambda d, s=s: s * d)


def apply_all(run, dep, out, h, pitch, zmax=10.0, preview_every=1):
    """Write calibrated depth for every frame that has network depth."""
    from multi_sg.rgbd import depth_millimeters
    (out / "depth_mm").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(exist_ok=True)
    turning = run.turning_mask()
    have = [int(i) for i in run.idx if (dep / f"{int(i):06d}.png").exists()]
    models, rows = {}, {}
    for i in have:                                   # pass 1: fit where lidar is trustworthy
        k = run._pos[i]
        if turning[k]:
            continue
        f = run.frame(i, load_rgb=False)
        f.rgb = np.zeros((480, 640, 3), np.uint8)
        r = fit_full(f, load_depth(dep / f"{i:06d}.png"), h, pitch)
        if r:
            models[i] = r
    good = np.array(sorted(models))
    if good.size == 0:
        raise SystemExit("no frame could be calibrated")
    for n, i in enumerate(have):                     # pass 2: apply (borrow nearest if needed)
        src = i if i in models else int(good[np.argmin(np.abs(good - i))])
        name, fn_ = models[src]
        d = load_depth(dep / f"{i:06d}.png")
        z = fn_(d).astype(np.float32)
        z[~np.isfinite(z) | (z <= 0.05) | (z > zmax)] = np.nan
        cv2.imwrite(str(out / "depth_mm" / f"{i:06d}.png"), depth_millimeters(np.nan_to_num(z, nan=0.0)))
        rows[i] = dict(frame=i, method=name, fit_from=src, borrowed=int(src != i),
                       turning=int(turning[run._pos[i]]),
                       median_m=float(np.nanmedian(z)), raw_median_m=float(np.nanmedian(d)))
        if preview_every and n % preview_every == 0:
            rgb = cv2.imread(str(run.root / "frames" / run.files[run._pos[i]]))
            t = np.clip(255 * (1 - np.nan_to_num(z, nan=zmax) / 5.0), 0, 255).astype(np.uint8)
            col = cv2.applyColorMap(t, cv2.COLORMAP_TURBO)
            col[~np.isfinite(z)] = 0
            pv = np.hstack([rgb, col])
            label = (f"frame {i}  calibrated depth (red 0 m -> blue 5 m)  {name}"
                     f"{'  [borrowed from ' + str(src) + ']' if src != i else ''}")
            for col_, th_ in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(pv, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_, th_, cv2.LINE_AA)
            cv2.imwrite(str(out / "preview" / f"{i:06d}.jpg"), pv, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(out / "calib_params.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values())).keys()))
        w.writeheader()
        w.writerows(rows[i] for i in have)
    nb = sum(r["borrowed"] for r in rows.values())
    print(f"wrote {len(rows)} calibrated depth maps to {out}  ({len(models)} fitted, {nb} borrowed)")
    from collections import Counter
    print("methods:", dict(Counter(models[i][0] for i in models)))


if __name__ == "__main__":
    main()
