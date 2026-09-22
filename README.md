# MobileNetV2 · Vision Lab

**See what the network sees.** A local-first, interactive visual laboratory for MobileNetV2, remastered from the original Streamlit visualization project.

Upload an image, follow its convolutional representations, inspect real weights, and generate class-specific Grad-CAM explanations. The main interface is now a responsive, dependency-free browser client backed by FastAPI and PyTorch. There is no JavaScript build step, paid API, external font, or analytics service.

## Run locally

Use Python 3.11–3.13. Create an isolated environment before installing dependencies.

```bash
git clone https://github.com/Duy1230/MobileNetv2-Visualization.git
cd MobileNetv2-Visualization
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install and start:

```bash
python -m pip install -r requirements.txt
python interface.py
```

Open **http://localhost:8501**. The server binds to loopback by default. `python server.py` is an equivalent entry point. The first trained-model run downloads the official weights; later runs reuse the Torch cache.

For a smaller CPU-only installation, install the matching CPU packages **before** the remaining requirements:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA, install an appropriate matching PyTorch/TorchVision pair using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then set `MNV2_DEVICE=cuda`. The application does not silently switch devices when a requested accelerator is unavailable.

### No-network demonstration

On Linux/macOS:

```bash
MNV2_WEIGHTS=untrained python interface.py
```

On Windows PowerShell:

```powershell
$env:MNV2_WEIGHTS = "untrained"
python interface.py
```

This is a **real MobileNetV2 with deterministic random weights**, not cached or fabricated predictions. It demonstrates the machinery without downloading weights. A persistent warning identifies untrained outputs in the interface and exports. Class scores and explanations in this mode have **no predictive meaning**.

To use trained inference without network access, pre-populate the Torch cache on the machine or set `MNV2_CHECKPOINT` to a trusted, matching TorchVision ImageNet MobileNetV2 state dictionary. Loading uses `weights_only=True` and strict shape/key validation. You are responsible for the checkpoint's provenance and its correspondence to the selected V1/V2 preprocessing and ImageNet labels. Arbitrary custom classifiers are not supported.

## Explore

**Feature maps.** Browse every convolution discovered from the running model—not a hard-coded subset. The atlas paginates at 32 channels, sorts by channel index, mean absolute response, or variance, and offers per-channel or layer-wide normalization. Click a channel for raw statistics and a nearest-neighbor enlargement; download its displayed PNG.

**Kernel studio.** Inspect a selected input-channel slice of spatial kernels, including the stem's RGB slices and depthwise filters. A 1×1 convolution is correctly represented as a full channel-mixing matrix, with output channels on rows and input channels on columns. It is not misleadingly rendered as an RGB or spatial image.

**Explain.** Select any of 1,000 ImageNet target classes and compute Grad-CAM on the last convolution. See the exact model input, class overlay, and heatmap together. Export the overlay as PNG. A zero-positive-attribution result is explicitly identified, not replaced by an artificial hotspot.

**Architecture.** Navigate 19 feature stages and all 52 convolutions. Inspect shapes, groups, kernels, strides, parameters, residual availability, and multiply-accumulate counts measured from the instantiated model. The standard 224×224 network contains 3,504,872 parameters and 300,774,272 Conv2d/Linear MACs by this implementation's accounting convention.

**Export.** Download a ZIP containing `report.json`, a full raw `activations.npy` tensor for the selected convolution, all 1,000 probabilities in `predictions.csv`, and the processed `model-input.png`. Arrays are float32, without pickle. The archive contains the 224×224 model crop, not the original uploaded file. Do not publish a report containing a private image without reviewing it.

Three original procedural diagnostics—Shapes, Stripes, and Spectrum—work without fetching any image assets. These are useful for spatial-response exploration, not an accuracy benchmark.

## Scientific interpretation

The default trained preset is `MobileNet_V2_Weights.IMAGENET1K_V2`; V1 remains selectable. Preprocessing and label names come from the **same weight enum**. V2 resizes the shorter edge to 232; V1 uses 256; both center-crop to 224 and apply the weight-specific normalization. The displayed input is reconstructed from the actual normalized tensor, so explanations align with what the network received.

Feature maps are **raw convolution outputs, before batch normalization and activation**, not post-ReLU features. They can be negative. Per-channel display normalization reveals structure but erases between-channel amplitude differences. Entire-layer scaling preserves the shared scale. Exported arrays are never display-normalized. Display palettes are linearly interpolated color ramps, not scientific measurements.

Grad-CAM uses the target **logit**, spatially averaged activation gradients, a channel-weighted sum, ReLU, and bilinear upsampling from the selected layer. The default final convolution produces a 7×7 map, so the enlarged explanation is coarse. It is not segmentation, a calibrated uncertainty estimate, or a causal proof. Softmax scores likewise are not calibrated confidence, especially for images outside ImageNet's distribution.

One MAC means one multiply-accumulate, not two FLOPs. Counts include convolution and linear layers, not normalization, activation, pooling, or data movement. The interface's timing is one server-side instrumented forward run, including copying logits back to CPU; it is not a rigorous throughput benchmark.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MNV2_WEIGHTS` | `IMAGENET1K_V2` | Initial UI preset: V2, V1, or `untrained`. |
| `MNV2_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps`. |
| `MNV2_NUM_THREADS` | Up to 4 | Positive PyTorch CPU thread count. |
| `MNV2_CHECKPOINT` | Unset | Trusted local, matching ImageNet state dict. Ignored in untrained mode. |
| `MNV2_HOST` | `127.0.0.1` | Server bind interface. |
| `PORT` | `8501` | Server port. |
| `MNV2_ALLOWED_HOSTS` | Local/test hosts | Comma-separated accepted hostnames. Include your domain when deploying. |
| `TORCH_HOME` | PyTorch default | Location of the model-weight cache. |

`.env.example` documents these variables. It is **not automatically loaded**; set the environment in your shell or deployment configuration.

### Docker

```bash
docker build -t mobilenet-vision-lab .
docker run --rm -p 127.0.0.1:8501:8501 \
  -v mobilenet-cache:/home/lab/.cache/torch \
  mobilenet-vision-lab
```

The container runs as a non-root user, installs CPU wheels, and exposes a health check. To avoid weight downloads for a demo, add `-e MNV2_WEIGHTS=untrained`. Docker packaging is supplied but was not built in the remaster's restricted test environment.

## Privacy, concurrency, and deployment

Images are processed **on the server running this application**, not in the browser. If you deploy remotely, they travel to that server. There is no third-party inference API, analytics, or image persistence to disk. Model weights alone are cached on disk. Each image gets an unguessable, session-specific capability token. There is no session-list endpoint.

The in-memory store holds at most eight analyses, with a 30-minute inactivity TTL; expiry is enforced on subsequent store operations. Oldest inactive/LRU sessions can be evicted sooner when the store is full. The UI's clear-session action deletes the server entry and clears retained browser references. An in-flight request can finish using an analysis it already obtained; deletion is not a secure-memory-erasure guarantee.

Model initialization is lazy and serialized. Each cached model has a reentrant lock covering inference and hook use, preventing cross-request hook contamination. Hooks are removed in `finally` blocks even on exceptions. Models run in evaluation mode; parameters are frozen. Grad-CAM uses input autograd and does not accumulate parameter gradients. Captured outputs are cloned to prevent later in-place operations changing them.

Uploads are limited to 10 MiB and 20 megapixels, restricted to still JPEG/PNG/WebP, EXIF-oriented, and alpha-composited onto white. Reads are bounded. API responses are `no-store`; host validation, same-origin write checks, and browser security headers are enabled. These are defensive defaults, **not a substitute for authentication, rate limits, TLS, or operational hardening**.

Use **one server worker**: analyses are intentionally process-local. Do not expose an unauthenticated instance to the public internet. For shared use, put authentication and rate/concurrency limits at a reverse proxy and review memory retention, trusted hosts, and upload policy. Additional authenticated users can otherwise exhaust anonymous session capacity.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --check static/app.js  # Optional JS syntax check; Node is not a runtime dependency.
python -m playwright install chromium
python tests/browser_smoke.py
```

Unit/API tests do not download pretrained weights. They cover preprocessing, metadata, channel sorting and pagination, upload validation, EXIF/alpha handling, hook cleanup, concurrency, session deletion/expiry, numerical exports, and Grad-CAM against an analytical reference. The Chromium smoke script exercises the actual interface against a live local server, checks several viewport widths, and saves screenshots in `test-artifacts/`.

Set `CHROMIUM_PATH` to an existing Chromium executable when needed. `MNV2_BROWSER_BRIDGE=1` is a restricted-environment test option: render the same HTML/CSS/JS from memory and bridge its fetch calls to the live local backend from Python, leaving browser navigation policy unchanged. It is not equivalent to directly verifying browser HTTP navigation or CSP enforcement. CI uses the normal direct-HTTP path.

See [the validation record](docs/VALIDATION.md) for what was actually run and what remains environment-dependent. CI runs backend tests on Python 3.11/3.12 and a Chromium browser smoke test without model downloads.

## Code map and migration

| File | Responsibility |
| --- | --- |
| `engine.py` | Image validation, weights, preprocessing, profiling, hooks, kernels, Grad-CAM. |
| `server.py` | FastAPI routes, bounded session store, exports, security defaults. |
| `static/` | Responsive, accessible vanilla HTML/CSS/JavaScript interface. |
| `interface.py` | Browser-app entry point. |
| `app.py` | Legacy notebook helper API. |
| `tests/` | Numerical, API, and browser tests. |

**Migration from the original:** launch with `python interface.py`, not `streamlit run interface.py`. Streamlit is no longer required. Existing notebook helpers (`read_image`, `load_mobileNet`, `normalize_filter`, `get_layer_by_name`, `preprocess_image`, `get_multiple_intermediate_outputs`, and `show_image`) remain exported from `app.py`. The legacy model/preprocess helpers retain the original V1 pairing. Install `matplotlib` separately only for the legacy `show_image` helper. The original `requirement.txt` name remains a forwarding alias for `requirements.txt`. ImageNet labels in the new application come from TorchVision metadata, not a working-directory-dependent text file.

## References

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [TorchVision MobileNetV2 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

Original project by **Duy1230**. The repository's original licensing status is unchanged; this remaster does not grant new license rights over third-party code or model weights.
