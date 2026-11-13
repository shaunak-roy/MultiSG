"""Export timestamped RGB-D pairs from a recorded video."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def depth_millimeters(depth):
    """Convert meters to uint16 millimeters; zero means invalid."""
    meters = np.asarray(depth, dtype=np.float64)
    rounded = np.rint(meters * 1000.0)

    valid = (
        np.isfinite(rounded)
        & (rounded >= 1)
        & (rounded <= 65535)
    )

    output = np.zeros(rounded.shape, dtype=np.uint16)
    output[valid] = rounded[valid].astype(np.uint16)
    return output


def write_image(path, image):
    import cv2

    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def export_video(
    video: Path,
    output: Path,
    predictor,
    *,
    stride=1,
    max_frames=None,
    save_mm=False,
    preview=False,
):
    import av
    import cv2

    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive")

    if not video.is_file():
        raise ValueError(f"Video file not found: {video}")

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("Output must be a new or empty directory")

    output.mkdir(parents=True, exist_ok=True)

    folders = ["rgb", "depth_m"]
    if save_mm:
        folders.append("depth_mm")
    if preview:
        folders.append("preview")

    for folder in folders:
        (output / folder).mkdir()

    metadata = {
        "schema_version": 1,
        "status": "running",
        "video": str(video.resolve()),
        "depth_units": "meters",
        "depth_dtype": "float32",
        "invalid_depth": 0,
        "depth_convention": "camera optical-axis Z",
        "rgb_depth_alignment": "same pixels",
        "geometry": (
            "Original decoded pixels; no undistortion "
            "or camera calibration applied"
        ),
        "timestamp_basis": (
            "Video presentation timestamp: PTS * time_base; "
            "not wall clock"
        ),
        "stride": stride,
        "frames_written": 0,
        "model": predictor.metadata,
        "depth_png": {
            "units": "millimeters",
            "dtype": "uint16",
            "invalid": 0,
            "meters_per_unit": 0.001,
            "maximum_m": 65.535,
        } if save_mm else None,
    }

    metadata_path = output / "run.json"

    def save_metadata():
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    save_metadata()

    try:
        with av.open(str(video)) as container:
            if not container.streams.video:
                raise ValueError("Input has no video stream")

            stream = container.streams.video[0]
            metadata["video_stream_index"] = stream.index
            metadata["average_fps"] = (
                float(stream.average_rate)
                if stream.average_rate else None
            )

            dimensions = None

            with (output / "frames.jsonl").open("w") as manifest:
                for frame_index, frame in enumerate(
                    container.decode(stream)
                ):
                    if frame_index % stride:
                        continue

                    bgr = frame.to_ndarray(format="bgr24")

                    if dimensions is None:
                        dimensions = bgr.shape[:2]
                        metadata.update(
                            height=dimensions[0],
                            width=dimensions[1],
                        )
                        save_metadata()
                    elif bgr.shape[:2] != dimensions:
                        raise ValueError(
                            "Video resolution changed; split it "
                            "into fixed-resolution clips"
                        )

                    start = time.perf_counter()

                    depth = np.asarray(
                        predictor.predict(bgr),
                        dtype=np.float32,
                    )

                    if depth.shape != bgr.shape[:2]:
                        raise ValueError(
                            "Depth must match the original "
                            "RGB frame resolution"
                        )

                    # Preserve numerical meters; never normalize this array.
                    depth = np.where(
                        np.isfinite(depth) & (depth > 0),
                        depth,
                        0,
                    ).astype(np.float32)

                    inference_ms = (
                        time.perf_counter() - start
                    ) * 1000

                    if not np.any(depth):
                        raise ValueError(
                            f"Frame {frame_index} has no valid depth"
                        )

                    name = f"{frame_index:08d}"
                    rgb_path = f"rgb/{name}.png"
                    depth_path = f"depth_m/{name}.npy"

                    write_image(output / rgb_path, bgr)
                    np.save(
                        output / depth_path,
                        depth,
                        allow_pickle=False,
                    )

                    timestamp = (
                        float(frame.pts * frame.time_base)
                        if frame.pts is not None
                        and frame.time_base is not None
                        else None
                    )

                    record = {
                        "frame_index": frame_index,
                        "pts": frame.pts,
                        "time_base": (
                            str(frame.time_base)
                            if frame.time_base is not None else None
                        ),
                        "timestamp_s": timestamp,
                        "rgb": rgb_path,
                        "depth_m": depth_path,
                        "inference_ms": inference_ms,
                    }

                    if save_mm:
                        mm_path = f"depth_mm/{name}.png"
                        write_image(
                            output / mm_path,
                            depth_millimeters(depth),
                        )
                        record["depth_mm"] = mm_path

                    if preview:
                        # Visualization only: fixed model-wide depth range.
                        normalized = np.clip(
                            depth / predictor.metadata["max_depth_m"],
                            0,
                            1,
                        )
                        colored = cv2.applyColorMap(
                            (normalized * 255).astype(np.uint8),
                            cv2.COLORMAP_INFERNO,
                        )
                        colored[depth == 0] = 0

                        write_image(
                            output / f"preview/{name}.png",
                            np.hstack([bgr, colored]),
                        )

                    # Record only fully exported pairs.
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()

                    metadata["frames_written"] += 1

                    print(
                        f"Frame {frame_index}: "
                        f"{inference_ms:.0f} ms",
                        flush=True,
                    )

                    if (
                        max_frames is not None
                        and metadata["frames_written"] >= max_frames
                    ):
                        break

            if metadata["frames_written"] == 0:
                raise ValueError("No video frames could be decoded")

        metadata["status"] = "complete"

    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        raise

    finally:
        save_metadata()

    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)

    parser.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl"],
        default="vits",
    )
    parser.add_argument(
        "--dataset",
        choices=["hypersim", "vkitti"],
        default="hypersim",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-mm", action="store_true")
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args(argv)

    if args.input_size < 14:
        parser.error("input-size must be at least 14")
    if args.stride < 1:
        parser.error("stride must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if not args.video.is_file():
        parser.error(f"Video file not found: {args.video}")
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            parser.error("Output must be a new or empty directory")

    try:
        from multi_sg.metric_depth import MetricDepthPredictor

        predictor = MetricDepthPredictor(
            repo=args.model_repo,
            checkpoint=args.checkpoint,
            encoder=args.encoder,
            dataset=args.dataset,
            device=args.device,
            input_size=args.input_size,
        )

        result = export_video(
            video=args.video,
            output=args.output,
            predictor=predictor,
            stride=args.stride,
            max_frames=args.max_frames,
            save_mm=args.save_mm,
            preview=args.preview,
        )

    except KeyboardInterrupt:
        print(
            "Interrupted. If export started, completed pairs "
            "are listed in frames.jsonl."
        )
        return 130

    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"RGB-D export failed: {exc}\n")

    print(
        f"Saved {result['frames_written']} RGB-D pairs "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())