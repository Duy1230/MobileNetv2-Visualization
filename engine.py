"""Model inspection primitives. No web/UI imports and no import-time downloads."""
from __future__ import annotations

import base64
import io
import math
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import functional as TF

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
WEIGHTS = {w.name: w for w in MobileNet_V2_Weights}
PALETTES = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "ember": [(13, 8, 29), (76, 22, 94), (158, 47, 92), (229, 108, 68), (252, 246, 184)],
    "gray": [(0, 0, 0), (255, 255, 255)],
}


def read_image(source: bytes | BinaryIO | str | Path) -> Image.Image:
    """Validate before decoding; orient EXIF and composite alpha over white."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Choose a non-empty image smaller than 10 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Only JPEG, PNG, and WebP images are supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 20-megapixel limit.")
                if getattr(image, "is_animated", False):
                    raise ValueError("Choose a still image, not an animation.")
                image.load()
                image = ImageOps.exif_transpose(image)
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                return Image.alpha_composite(background, rgba).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This image is damaged, unsupported, or too large.") from exc


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


def normalize(values: np.ndarray, low: float | None = None,
              high: float | None = None) -> np.ndarray:
    """Finite, zero-safe display normalization; never changes exported tensors."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0, posinf=0, neginf=0)
    low = float(values.min()) if low is None else low
    high = float(values.max()) if high is None else high
    if high - low < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def colorize(values: np.ndarray, palette: str = "viridis") -> Image.Image:
    if palette not in PALETTES:
        raise ValueError("Unknown palette.")
    anchors = np.array(PALETTES[palette], dtype=float)
    points = np.linspace(0, 1, len(anchors))
    values = np.clip(values, 0, 1)
    rgb = np.stack([np.interp(values, points, anchors[:, i]) for i in range(3)], -1)
    return Image.fromarray(rgb.astype(np.uint8))


def sample_image(name: str) -> Image.Image:
    """Original procedural diagnostic inputs: no external assets or downloads."""
    size = 512
    y, x = np.mgrid[:size, :size]
    if name == "stripes":
        wave = ((np.sin(x / 12 + y / 30) + 1) / 2)[..., None]
        array = np.array([20, 38, 54]) + wave * np.array([210, 166, 83])
        return Image.fromarray(array.astype(np.uint8))
    if name == "spectrum":
        array = np.stack([x / 2, y / 2, (np.sin((x + y) / 52) + 1) * 100], -1)
        return Image.fromarray(array.astype(np.uint8))
    if name != "shapes":
        raise ValueError("Unknown sample.")
    image = Image.new("RGB", (size, size), "#e9e3d5")
    draw = ImageDraw.Draw(image)
    for r in range(210, 15, -4):
        v = (210 - r) / 195
        color = (int(17 + 36 * v), int(53 + 95 * v), int(60 + 40 * v))
        draw.ellipse((260 - r, 254 - r, 260 + r, 254 + r), fill=color)
    draw.rectangle((0, 356, 512, 512), fill="#c8c1ad")
    draw.ellipse((97, 362, 423, 424), fill="#ada792")
    draw.polygon([(89, 372), (196, 114), (303, 372)], fill="#e4af59")
    draw.polygon([(196, 114), (225, 320), (303, 372)], fill="#be7d42")
    draw.ellipse((265, 262, 404, 401), fill="#ec704e")
    draw.arc((273, 267, 400, 393), 200, 295, fill="#ffb58c", width=5)
    return image


@dataclass(frozen=True)
class Layer:
    name: str
    stage: int
    kind: str
    channels_in: int
    channels_out: int
    kernel: tuple[int, int]
    stride: tuple[int, int]
    groups: int
    parameters: int
    output: tuple[int, ...]
    macs: int


def get_layer(model: nn.Module, name: str) -> nn.Module:
    try:
        return model.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(f"Unknown layer: {name}") from exc


def capture_outputs(model: nn.Module, tensor: torch.Tensor,
                    names: list[str]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Validate all names first and always remove hooks, including on failure."""
    modules = [(name, get_layer(model, name)) for name in dict.fromkeys(names)]
    captured: dict[str, torch.Tensor] = {}
    handles = []
    try:
        for name, module in modules:
            def hook(_module, _args, output, key=name):
                # clone prevents subsequent in-place operations changing the snapshot.
                captured[key] = output.detach().cpu().clone()
            handles.append(module.register_forward_hook(hook))
        with torch.inference_mode():
            logits = model(tensor).detach().cpu().clone()
        return captured, logits
    finally:
        for handle in handles:
            handle.remove()


class Engine:
    """One immutable eval model and one lock shared by all its inference paths."""
    def __init__(self, weights: str = "IMAGENET1K_V2", device: str = "cpu",
                 checkpoint: str | None = None):
        if weights not in {*WEIGHTS, "untrained"}:
            raise ValueError("Unknown weight preset.")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Device must be cpu, cuda, or mps.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable.")
        self.lock = threading.RLock()
        self.weights_name = weights
        self.preset = WEIGHTS.get(weights, MobileNet_V2_Weights.IMAGENET1K_V2)
        self.trained = weights != "untrained"
        self.device = torch.device(device)
        # fork_rng restores global RNG state, keeping the demo reproducible.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = mobilenet_v2(
                weights=self.preset if self.trained and not checkpoint else None
            )
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        self.checkpoint = bool(checkpoint)
        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                module.inplace = False
        self.model.eval().requires_grad_(False).to(self.device)
        self.categories = self.preset.meta["categories"]
        self.transform = self.preset.transforms()
        self.layers, self.stages = self._profile()
        self.layer_names = {layer.name for layer in self.layers}

    def _profile(self) -> tuple[list[Layer], list[dict]]:
        layers, stages, handles = [], [], []
        linear_macs = 0
        def conv_hook(name, module):
            def hook(_module, _inputs, output):
                shape = tuple(output.shape[1:])
                kind = "depthwise" if module.groups == module.in_channels else (
                    "pointwise" if module.kernel_size == (1, 1) else "spatial")
                macs = math.prod(shape) * (module.in_channels // module.groups) * math.prod(module.kernel_size)
                layers.append(Layer(name, int(name.split(".")[1]), kind,
                                    module.in_channels, module.out_channels,
                                    module.kernel_size, module.stride, module.groups,
                                    sum(p.numel() for p in module.parameters()), shape, macs))
            return hook
        def linear_hook(module, _inputs, _output):
            nonlocal linear_macs
            linear_macs += module.in_features * module.out_features
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    handles.append(module.register_forward_hook(conv_hook(name, module)))
                elif isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(linear_hook))
            for index, block in enumerate(self.model.features):
                def stage_hook(module, _inputs, output, i=index):
                    stages.append({"index": i, "name": f"features.{i}",
                                   "output": list(output.shape[1:]),
                                   "parameters": sum(p.numel() for p in module.parameters()),
                                   "residual": bool(getattr(module, "use_res_connect", False))})
                handles.append(block.register_forward_hook(stage_hook))
            with torch.inference_mode():
                self.model(torch.zeros(1, 3, 224, 224, device=self.device))
        finally:
            for handle in handles:
                handle.remove()
        self.macs = sum(layer.macs for layer in layers) + linear_macs
        for stage in stages:
            stage["macs"] = sum(layer.macs for layer in layers if layer.stage == stage["index"])
        return layers, stages

    def metadata(self) -> dict:
        return {"model": "MobileNetV2", "weights": self.weights_name,
                "trained": self.trained, "local_checkpoint": self.checkpoint,
                "device": str(self.device), "parameters": sum(p.numel() for p in self.model.parameters()),
                "macs": self.macs, "classes": len(self.categories),
                "layers": [asdict(layer) for layer in self.layers], "stages": self.stages,
                "preprocessing": {"resize": self.transform.resize_size,
                                  "crop": self.transform.crop_size,
                                  "mean": self.transform.mean, "std": self.transform.std},
                "torch": torch.__version__}

    def prepare(self, image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
        tensor = self.transform(image).unsqueeze(0)
        mean = torch.tensor(self.transform.mean).view(3, 1, 1)
        std = torch.tensor(self.transform.std).view(3, 1, 1)
        crop = TF.to_pil_image((tensor[0] * std + mean).clamp(0, 1))
        return tensor, crop

    def predict(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        with self.lock:
            value = tensor.to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _, logits = capture_outputs(self.model, value, [])
            elapsed = (time.perf_counter() - start) * 1000
            return logits[0].softmax(0), elapsed

    def activation(self, tensor: torch.Tensor, name: str) -> np.ndarray:
        self.validate_layer(name)
        with self.lock:
            outputs, _ = capture_outputs(self.model, tensor.to(self.device), [name])
        return outputs[name][0].numpy()

    def validate_layer(self, name: str) -> None:
        if name not in self.layer_names:
            raise ValueError("Select a convolutional layer from the architecture explorer.")

    def filters(self, name: str, input_channel: int = 0) -> np.ndarray:
        self.validate_layer(name)
        with self.lock:
            values = get_layer(self.model, name).weight.detach().cpu().numpy().copy()
        if not 0 <= input_channel < values.shape[1]:
            raise ValueError("Input-channel index is outside this layer's weight tensor.")
        return values[:, input_channel]

    def mixing_matrix(self, name: str) -> np.ndarray:
        self.validate_layer(name)
        with self.lock:
            values = get_layer(self.model, name).weight.detach().cpu().numpy().copy()
        return values.mean(axis=(2, 3))

    def gradcam(self, tensor: torch.Tensor, name: str, target: int) -> np.ndarray:
        self.validate_layer(name)
        if not 0 <= target < len(self.categories):
            raise ValueError("Class index must be between 0 and 999.")
        with self.lock, torch.enable_grad():
            captured = []
            handle = get_layer(self.model, name).register_forward_hook(
                lambda _m, _i, output: captured.append(output))
            try:
                value = tensor.to(self.device).clone().requires_grad_(True)
                logits = self.model(value)
                gradients, = torch.autograd.grad(logits[0, target], captured[0])
                cam = (gradients.mean((2, 3), keepdim=True) * captured[0]).sum(1).relu()
                cam = torch.nn.functional.interpolate(cam[:, None], size=(224, 224),
                                                       mode="bilinear", align_corners=False)
                return normalize(cam[0, 0].detach().cpu().numpy())
            finally:
                handle.remove()


def channel_page(values: np.ndarray, page: int = 0, per_page: int = 32,
                 sort: str = "energy", normalization: str = "channel",
                 palette: str = "viridis") -> dict:
    if values.ndim != 3 or not 1 <= per_page <= 64:
        raise ValueError("Expected C x H x W data and a page size from 1 to 64.")
    if sort not in {"energy", "index", "variance"} or normalization not in {"channel", "layer"}:
        raise ValueError("Invalid display option.")
    if palette not in PALETTES:
        raise ValueError("Unknown palette.")
    total = len(values)
    if not 0 <= page < math.ceil(total / per_page):
        raise ValueError("Page is outside the channel range.")
    energy = np.abs(values).mean((1, 2))
    variance = values.var((1, 2))
    order = np.arange(total) if sort == "index" else np.argsort(
        -(energy if sort == "energy" else variance), kind="stable")
    bounds = (float(values.min()), float(values.max())) if normalization == "layer" else (None, None)
    channels = []
    for i in order[page * per_page:(page + 1) * per_page]:
        value = values[i]
        channels.append({"index": int(i), "mean": float(value.mean()),
                         "energy": float(energy[i]), "std": float(value.std()),
                         "min": float(value.min()), "max": float(value.max()),
                         "zero_fraction": float((np.abs(value) < 1e-8).mean()),
                         "image": data_url(colorize(normalize(value, *bounds), palette))})
    return {"channels": channels, "total": total, "page": page,
            "pages": math.ceil(total / per_page), "shape": list(values.shape),
            "range": [float(values.min()), float(values.max())],
            "mean_energy": float(energy.mean()), "normalization": normalization,
            "palette": palette, "sort": sort}
