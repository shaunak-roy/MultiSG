import subprocess, numpy as np, cv2

W, H = 640, 480
cmd = ["ffmpeg", "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
       "-i", "tcp://67.194.38.225:8888",
       "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
size = W * H * 3

try:
    while True:
        buf = bytearray()
        while len(buf) < size:
            chunk = proc.stdout.read(size - len(buf))
            if not chunk:
                raise SystemExit("stream ended")
            buf += chunk
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()

        # --- your processing here ---

        cv2.imshow("mbot", frame)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    proc.terminate()
    cv2.destroyAllWindows()