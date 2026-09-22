"""Local-first HTTP application. Run: python interface.py"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import secrets
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from engine import (MAX_BYTES, WEIGHTS, Engine, channel_page, colorize, data_url,
                    normalize, png_bytes, read_image, sample_image)

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = os.getenv("MNV2_WEIGHTS", "IMAGENET1K_V2")
DEVICE = os.getenv("MNV2_DEVICE", "cpu")
logger = logging.getLogger(__name__)


@dataclass
class Analysis:
    engine: Engine
    tensor: torch.Tensor
    crop: Image.Image
    probabilities: torch.Tensor
    elapsed_ms: float
    source: str
    original_size: tuple[int, int]


class SessionStore:
    """Bounded, process-local, expiring capability-token store. Never writes images."""
    def __init__(self, capacity: int = 8, ttl: float = 1800):
        self.capacity, self.ttl = capacity, ttl
        self.items: OrderedDict[str, tuple[float, Analysis]] = OrderedDict()
        self.lock = threading.RLock()

    def _expire(self):
        now = time.monotonic()
        for key, (touched, _) in list(self.items.items()):
            if now - touched >= self.ttl:
                del self.items[key]

    def put(self, analysis: Analysis) -> str:
        with self.lock:
            self._expire()
            while len(self.items) >= self.capacity:
                self.items.popitem(last=False)
            key = secrets.token_urlsafe(24)
            self.items[key] = (time.monotonic(), analysis)
            return key

    def get(self, key: str) -> Analysis:
        with self.lock:
            self._expire()
            if key not in self.items:
                raise HTTPException(404, "This session expired. Analyze your image again.")
            _, analysis = self.items.pop(key)
            self.items[key] = (time.monotonic(), analysis)
            return analysis

    def delete(self, key: str):
        with self.lock:
            self.items.pop(key, None)


store = SessionStore()
engines: dict[str, Engine] = {}
engine_lock = threading.RLock()
ingest_lock = threading.BoundedSemaphore(2)


def get_engine(weights: str) -> Engine:
    if weights not in {*WEIGHTS, "untrained"}:
        raise ValueError("Unknown weight preset.")
    with engine_lock:
        if weights not in engines:
            try:
                engines[weights] = Engine(weights, DEVICE,
                    checkpoint=os.getenv("MNV2_CHECKPOINT") if weights != "untrained" else None)
            except Exception as exc:
                logger.exception("Model initialization failed")
                raise HTTPException(503,
                    "Could not load model weights. Check your connection/device or select "
                    "Offline demo (untrained). For offline trained inference, configure "
                    "MNV2_CHECKPOINT with a trusted matching TorchVision state dict.") from exc
        return engines[weights]


def prediction_rows(analysis: Analysis, count: int = 5) -> list[dict]:
    scores, indices = analysis.probabilities.topk(count)
    return [{"index": int(i), "label": analysis.engine.categories[int(i)],
             "probability": float(score)} for score, i in zip(scores, indices)]


def summary(key: str, analysis: Analysis) -> dict:
    p = analysis.probabilities
    return {"id": key, "model": analysis.engine.metadata(),
            "predictions": prediction_rows(analysis), "crop": data_url(analysis.crop),
            "source": analysis.source, "original_size": analysis.original_size,
            "elapsed_ms": analysis.elapsed_ms,
            "entropy_bits": float(-(p * p.clamp_min(1e-20).log2()).sum()),
            "warning": None if analysis.engine.trained else
                "UNTRAINED DEMO: activations use random weights; class scores and explanations have no predictive meaning."}


def make_analysis(raw: bytes, sample: str | None, weights: str) -> dict:
    with ingest_lock:
        image = sample_image(sample) if sample else read_image(raw)
        engine = get_engine(weights)
        tensor, crop = engine.prepare(image)
        probabilities, elapsed = engine.predict(tensor)
        analysis = Analysis(engine, tensor, crop, probabilities, elapsed,
                            f"Synthetic diagnostic: {sample}" if sample else "Uploaded image",
                            image.size)
        return summary(store.put(analysis), analysis)


app = FastAPI(title="MobileNetV2 Vision Lab", version="2.0.0", docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.getenv(
    "MNV2_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver").split(","))


@app.middleware("http")
async def security_headers(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
        if origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin writes are not allowed."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif request.url.path == "/" or request.url.path.endswith(".html"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
    return response


@app.exception_handler(ValueError)
async def invalid_input(_request: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0", "default_weights": DEFAULT_WEIGHTS,
            "device": DEVICE, "ready_models": list(engines)}


@app.get("/api/classes")
def classes():
    return WEIGHTS["IMAGENET1K_V2"].meta["categories"]


@app.post("/api/analyze")
async def analyze(request: Request, sample: str | None = None,
                  weights: str = DEFAULT_WEIGHTS):
    # Read bounded chunks instead of accepting an unbounded multipart/body allocation.
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds the 10 MiB upload limit.")
        raw.extend(chunk)
    if sample and raw:
        raise ValueError("Choose either an upload or a diagnostic sample, not both.")
    return await run_in_threadpool(make_analysis, bytes(raw), sample, weights)


@app.delete("/api/analysis/{key}", status_code=204)
def delete_analysis(key: str):
    store.delete(key)
    return Response(status_code=204)


@app.get("/api/analysis/{key}/activations")
def activations(key: str, layer: str = "features.0.0", page: int = Query(0, ge=0),
                per_page: int = Query(32, ge=1, le=64), sort: str = "energy",
                normalization: str = "channel", palette: str = "viridis"):
    analysis = store.get(key)
    values = analysis.engine.activation(analysis.tensor, layer)
    return channel_page(values, page, per_page, sort, normalization, palette)


@app.get("/api/analysis/{key}/filters")
def filters(key: str, layer: str = "features.0.0", page: int = Query(0, ge=0),
            input_channel: int = Query(0, ge=0), palette: str = "viridis"):
    analysis = store.get(key)
    engine = analysis.engine
    values = engine.filters(layer, input_channel)
    result = channel_page(values, page, 32, "index", "layer", palette)
    matrix = engine.mixing_matrix(layer)
    result["matrix"] = data_url(colorize(normalize(matrix), palette))
    result["matrix_shape"] = list(matrix.shape)
    result["matrix_range"] = [float(matrix.min()), float(matrix.max())]
    result["input_channels"] = next(l.channels_in // l.groups for l in engine.layers if l.name == layer)
    return result


@app.post("/api/analysis/{key}/cam")
def cam(key: str, target: int = Query(..., ge=0, le=999),
        layer: str = "features.18.0", alpha: float = Query(0.55, ge=0, le=1)):
    analysis = store.get(key)
    heat = analysis.engine.gradcam(analysis.tensor, layer, target)
    colored = colorize(heat, "ember")
    # Zero attribution preserves the input rather than painting a misleading tint.
    opacity = (heat * alpha)[..., None]
    blended = np.asarray(analysis.crop) * (1 - opacity) + np.asarray(colored) * opacity
    overlay = Image.fromarray(blended.clip(0, 255).astype(np.uint8))
    return {"overlay": data_url(overlay), "heatmap": data_url(colored),
            "class_index": target, "label": analysis.engine.categories[target],
            "layer": layer, "nonzero": bool(heat.max() > 0),
            "trained": analysis.engine.trained}


@app.get("/api/analysis/{key}/export")
def export(key: str, layer: str = "features.0.0"):
    analysis = store.get(key)
    values = analysis.engine.activation(analysis.tensor, layer)
    report = summary("omitted", analysis)
    report.pop("crop")
    report["export"] = {"layer": layer, "activation_shape": list(values.shape),
        "activation_dtype": str(values.dtype), "activation_point": "raw convolution output, before batch norm / activation",
        "privacy": "Includes the 224x224 processed input, not the original upload.",
        "mac_convention": "One multiply-accumulate is one MAC; Conv2d and Linear only.",
        "interpretation": "Softmax is not calibrated confidence; Grad-CAM is not a causal explanation."}
    array_file = io.BytesIO()
    np.save(array_file, values, allow_pickle=False)
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["class_index", "class_label", "probability"])
    for index, probability in enumerate(analysis.probabilities.tolist()):
        writer.writerow([index, analysis.engine.categories[index], probability])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("report.json", json.dumps(report, indent=2, allow_nan=False))
        output.writestr("activations.npy", array_file.getvalue())
        output.writestr("predictions.csv", table.getvalue())
        output.writestr("model-input.png", png_bytes(analysis.crop))
    return Response(archive.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="mobilenet-analysis.zip"'})


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


def main():
    import uvicorn
    threads = int(os.getenv("MNV2_NUM_THREADS", str(min(4, os.cpu_count() or 1))))
    if threads < 1:
        raise ValueError("MNV2_NUM_THREADS must be positive.")
    torch.set_num_threads(threads)
    # A single worker is required: sessions are deliberately process-local.
    uvicorn.run("server:app", host=os.getenv("MNV2_HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8501")), workers=1, access_log=False)


if __name__ == "__main__":
    main()
