# Remaster validation record

## Executed locally

- 49 pytest unit/API cases passed.
- Python source compilation and Node JavaScript syntax checks passed.
- Real Chromium rendered the interface and exercised image uploading, activations, layer switching, sorting, pagination, the inspector, pointwise matrices, Grad-CAM, PNG/ZIP downloads, architecture links, keyboard tab navigation, and clearing a session.
- Browser viewports: 1440, 768, and 390 CSS pixels. No page-level horizontal overflow or uncaught JavaScript exceptions were observed.
- Screenshots of the actual desktop, kernel, explanation, and mobile views were inspected.

The sandbox's browser navigation policy blocks all URL navigation, including loopback. Browser interaction testing therefore used the documented in-memory rendering / live-local-backend fetch bridge. This verifies the frontend logic, rendering, and real backend responses, but not direct browser-to-server HTTP navigation, browser CSP behavior, or a deployed reverse proxy. The included CI workflow uses the normal direct-HTTP smoke path.

## Environment

Python 3.13.5; PyTorch 2.10.0+cpu; TorchVision 0.25.0+cpu; FastAPI 0.128.2; Uvicorn 0.48.0; NumPy 2.3.5; Pillow 12.3.0; pytest 9.0.2; HTTPX 0.28.1; Playwright 1.57.0 and the available Chromium executable.

The requirements specify compatible ranges rather than locking every transitive dependency. The versions above are the actual test environment, not a claim to have tested all versions in those ranges.

## Important limits

The test environment could not download official pretrained weights. Tests used the real randomly initialized MobileNetV2 plus an analytically controlled toy model for Grad-CAM verification. No ImageNet accuracy claim or trained-model interpretation was validated here. Pretrained-load failure is tested and is never silently converted into an untrained result.

CUDA/MPS execution, Docker image construction, Windows-native execution, CI on remote Python 3.11/3.12, multi-user production hosting, and sustained load were not exercised locally. These are not described as verified. First-run model downloads and public deployment require an appropriately networked and secured environment.
