"""Capture calibration images from the RPi camera's live feed.

No image processing is done here - it just shows the stream and saves frames.
Run calibrate_camera.py on the saved folder afterwards to get K.

    python3 src/camera_utils/capture_images.py --out src/camera_utils/calib_images

    SPACE / c : save current frame
    q / ESC   : quit
"""

import argparse
import os
import subprocess
import threading

import cv2
import numpy as np

DEFAULT_URL = "tcp://67.194.38.225:8888"


def stream_frames(url, w, h, ffmpeg="ffmpeg"):
    """Yield the latest BGR frame from the ffmpeg TCP stream.

    A background thread drains the pipe and keeps only the newest frame, so
    the preview never lags behind the camera. `ffmpeg` is the executable to run.
    """
    cmd = [ffmpeg, "-fflags", "nobuffer", "-flags", "low_delay",
           "-probesize", "32", "-analyzeduration", "0",
           "-i", url, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)
    size = w * h * 3
    latest = {"frame": None, "id": 0, "done": False}
    cond = threading.Condition()

    def reader():
        try:
            while True:
                buf = bytearray()
                while len(buf) < size:
                    chunk = proc.stdout.read(size - len(buf))
                    if not chunk:
                        return
                    buf += chunk
                with cond:
                    latest["frame"], latest["id"] = buf, latest["id"] + 1
                    cond.notify()
        finally:
            with cond:
                latest["done"] = True
                cond.notify()

    threading.Thread(target=reader, daemon=True).start()
    seen = 0
    try:
        while True:
            with cond:
                cond.wait_for(lambda: latest["id"] != seen or latest["done"])
                if latest["id"] == seen:
                    return
                seen, buf = latest["id"], latest["frame"]
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.terminate()


def next_index(folder):
    nums = [int(f[6:9]) for f in os.listdir(folder)
            if f.startswith("calib_") and f[6:9].isdigit()]
    return max(nums, default=0) + 1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="ffmpeg stream URL")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=os.path.join(here, "calib_images"),
                    help="folder to save images into")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx = next_index(args.out)  # don't overwrite earlier sessions
    saved, flash = 0, 0
    print(f"Saving to {args.out}  (SPACE/c = save, q = quit)")

    for frame in stream_frames(args.url, args.width, args.height):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), ord("c")):
            path = os.path.join(args.out, f"calib_{idx:03d}.png")
            cv2.imwrite(path, frame)
            print("saved", path)
            idx, saved, flash = idx + 1, saved + 1, 3
        elif key in (ord("q"), 27):
            break

        view = frame
        if flash:  # brief white border as capture feedback (display only)
            view = frame.copy()
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1),
                          (255, 255, 255), 8)
            flash -= 1
        cv2.imshow("capture", view)
    else:
        print("stream ended")

    cv2.destroyAllWindows()
    print(f"Saved {saved} images this session.")


if __name__ == "__main__":
    main()
