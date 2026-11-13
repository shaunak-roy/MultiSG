"""Load Depth Anything V2 metric weights and predict depth in meters."""

import hashlib
import importlib
import inspect
import sys
from pathlib import Path


CONFIGS = {
    "vits": {
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

MAX_DEPTH = {
    "hypersim": 20.0,
    "vkitti": 80.0,
}


class MetricDepthPredictor:
    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        encoder: str = "vits",
        dataset: str = "hypersim",
        device: str = "auto",
        input_size: int = 518,
    ):
        import cv2
        import torch
        from torchvision.transforms import Compose

        # Import the metric implementation, not the relative-depth one.
        root = (repo.resolve() / "metric_depth").resolve()

        if not (root / "depth_anything_v2/dpt.py").is_file():
            raise ValueError(
                "--model-repo must point to a Depth-Anything-V2 checkout "
                "containing metric_depth/"
            )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        sys.path.insert(0, str(root))
        module = importlib.import_module("depth_anything_v2.dpt")

        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(
                "Another depth_anything_v2 implementation is already loaded. "
                "Run this in a fresh process."
            )

        model_type = module.DepthAnythingV2

        if "max_depth" not in inspect.signature(model_type).parameters:
            raise ValueError("Expected the metric-depth implementation")

        transforms = importlib.import_module(
            "depth_anything_v2.util.transform"
        )

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.model = model_type(
            encoder=encoder,
            **CONFIGS[encoder],
            max_depth=MAX_DEPTH[dataset],
        )

        # Use metric weights matching the selected encoder and dataset.
        # Strict loading checks shapes, not whether weights are metric.
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        self.transform = Compose([
            transforms.Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            transforms.NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.PrepareForNet(),
        ])

        with checkpoint.open("rb") as weights:
            checksum = hashlib.file_digest(weights, "sha256").hexdigest()

        self.metadata = {
            "model": "Depth Anything V2 metric",
            "encoder": encoder,
            "dataset": dataset,
            "max_depth_m": MAX_DEPTH[dataset],
            "device": str(self.device),
            "input_size": input_size,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checksum,
        }

    def predict(self, bgr):
        """Return an H×W float depth array in meters."""
        import cv2
        import torch
        import torch.nn.functional as F

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype("float32") / 255.0

        image = self.transform({"image": rgb})["image"]
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            depth = self.model(tensor)

            # Restore the original RGB frame's resolution.
            depth = F.interpolate(
                depth[:, None],
                size=bgr.shape[:2],
                mode="bilinear",
                align_corners=True,
            )

        return depth[0, 0].cpu().numpy()