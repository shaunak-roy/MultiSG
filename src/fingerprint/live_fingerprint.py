"""
Live object fingerprints from the MBot camera stream.

Loads DovSG's perception models once, then runs them on the newest frame from the
robot's ffmpeg TCP stream and shows the annotated result. Frames that arrive while a
frame is being processed are dropped, so the view lags by one processing step at most.

    python src/fingerprint/live_fingerprint.py --url tcp://<mbot-ip>:8888

    s       : save the current frame's annotated image + det_res .pkl
    q / ESC : quit
"""
import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE.parent / "camera_utils" / "calibration" / "calibration.npz"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "camera_utils" / "calibration"))
from capture_images import DEFAULT_URL, stream_frames  # noqa: E402
from fingerprint_image import load_semantic_memory  # noqa: E402


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # bundled binary, no admin install needed

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_undistorter(calib_path, width, height):
    """Undistort maps from the camera_utils calibration. The undistorted image keeps the
    same K, so detections on it can be back-projected with K and zero distortion."""
    calib = np.load(calib_path)
    K, dist = calib["camera_matrix"], calib["dist_coeffs"]
    if tuple(calib["image_size"]) != (width, height):
        sys.exit(f"{calib_path} is for {tuple(calib['image_size'])}, stream is {(width, height)}. "
                 "Recalibrate at the stream resolution or pass --no_undistort.")
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_16SC2)
    return K, dist, lambda img: cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="MBot ffmpeg stream URL")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--add_classes", nargs="*", default=[])
    parser.add_argument("--out", default="fingerprints/live", help="where 's' saves frames")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB), help="calibration .npz from calibrate_camera.py")
    parser.add_argument("--no_undistort", action="store_true", help="run on raw (distorted) frames")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()  # before load_semantic_memory changes directory
    ffmpeg = find_ffmpeg()
    K = dist = None
    undistort = lambda img: img  # noqa: E731
    if not args.no_undistort:
        K, dist, undistort = load_undistorter(Path(args.calib).resolve(), args.width, args.height)
        print(f"Undistorting with {args.calib}")

    print("Loading models (takes ~20 s)...")
    semantic_memory = load_semantic_memory(args.device, args.add_classes)
    print(f"Connecting to {args.url}  (s = save, q = quit)")

    got_frame, saved = False, 0
    for frame_bgr in stream_frames(args.url, args.width, args.height, ffmpeg=ffmpeg):
        got_frame = True
        frame_bgr = undistort(frame_bgr)
        start = time.perf_counter()
        with torch.no_grad():
            det_res, annotated, _ = semantic_memory.semantic_process(image=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        dt = time.perf_counter() - start

        view = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        labels = [det_res["classes"][i] for i in det_res["class_id"]]
        cv2.putText(view, f"{dt:.2f}s/frame  {len(labels)} objects", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("mbot fingerprints", view)
        print(f"{dt:.2f}s  {labels}")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out_dir.mkdir(parents=True, exist_ok=True)
            name = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(out_dir / f"{name}.jpg"), view)
            cv2.imwrite(str(out_dir / f"{name}_raw.png"), frame_bgr)
            with open(out_dir / f"{name}.pkl", "wb") as f:
                # K: intrinsics of the saved frame (undistorted, so zero distortion); None if --no_undistort
                pickle.dump({**det_res, "K": K}, f)
            saved += 1
            print(f"saved {out_dir / name}.*")

    if not got_frame:
        print(f"No frames from {args.url}. Is the MBot streaming, and is this PC on the same network?")
    cv2.destroyAllWindows()
    print(f"Saved {saved} frames.")


if __name__ == "__main__":
    main()
