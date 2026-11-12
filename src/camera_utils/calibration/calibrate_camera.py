"""Compute the camera matrix K and distortion from saved checkerboard images.

Capture images first with capture_images.py, then:
    python3 src/camera_utils/calibrate_camera.py --cols 9 --rows 6 --square 25

--cols/--rows are the number of INNER corners (a 10x7-square board is 9x6).
--images defaults to src/camera_utils/calib_images.
Results are written to --out as YAML (OpenCV FileStorage) plus a .npz copy.
"""

import argparse
import glob
import os

import cv2
import numpy as np

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def board_points(cols, rows, square):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def find_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return ok, corners


def calibrate(img_points, objp, image_size):
    obj_points = [objp] * len(img_points)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)
    per_view = []
    for op, ip, r, t in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((ip - proj) ** 2, axis=2)))))
    return rms, K, dist, per_view


def save(path, K, dist, image_size, rms, n_views, pattern, square):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("rms_reprojection_error", rms)
    fs.write("num_views", n_views)
    fs.write("board_cols", pattern[0])
    fs.write("board_rows", pattern[1])
    fs.write("square_size", square)
    fs.release()
    np.savez(os.path.splitext(path)[0] + ".npz", camera_matrix=K,
             dist_coeffs=dist, image_size=np.array(image_size), rms=rms)


def load_calibration(path):
    """Helper for other scripts: returns (camera_matrix, dist_coeffs, (w, h))."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    size = (int(fs.getNode("image_width").real()),
            int(fs.getNode("image_height").real()))
    fs.release()
    return K, dist, size


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(here, "calib_images"),
                    help="folder of checkerboard images")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square", type=float, default=25.0,
                    help="square side length (units of your choice, e.g. mm)")
    ap.add_argument("--min-views", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(here, "calibration.yaml"))
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = board_points(args.cols, args.rows, args.square)

    paths = sorted(p for ext in ("png", "jpg", "jpeg", "bmp")
                   for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    img_points, used, image_size = [], [], None
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip {p}: unreadable")
            continue
        if image_size is None:
            image_size = gray.shape[::-1]
        elif gray.shape[::-1] != image_size:
            print(f"skip {p}: size mismatch")
            continue
        ok, corners = find_corners(gray, pattern)
        print(f"{'ok  ' if ok else 'miss'} {os.path.basename(p)}")
        if ok:
            img_points.append(corners)
            used.append(os.path.basename(p))

    if len(img_points) < args.min_views:
        raise SystemExit(f"Board found in {len(img_points)} images; need at least "
                         f"{args.min_views}. Check --cols/--rows (inner corners).")

    rms, K, dist, per_view = calibrate(img_points, objp, image_size)
    np.set_printoptions(precision=4, suppress=True)
    print(f"\nCalibrated on {len(img_points)}/{len(paths)} images, "
          f"RMS reprojection error = {rms:.4f} px")
    print("K (camera_matrix):\n", K)
    print("dist_coeffs (k1 k2 p1 p2 k3):\n", dist.ravel())
    worst = sorted(zip(used, per_view), key=lambda x: -x[1])[:3]
    print("worst images:", ", ".join(f"{n}: {e:.3f}px" for n, e in worst))
    if rms > 1.0:
        print("WARNING: RMS > 1 px - check square size, board flatness, or "
              "delete blurry images listed above and re-run.")
    save(args.out, K, dist, image_size, rms, len(img_points), pattern, args.square)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
