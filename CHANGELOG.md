# Changelog

## 2.0.0 — Vision Lab remaster

Replaced the monolithic Streamlit UI with a responsive local web laboratory: FastAPI, PyTorch, and vanilla HTML/CSS/JavaScript. Added complete convolution discovery, stage navigation, activation pagination and statistics, raw tensor/report downloads, a proper pointwise channel-mixing view, class-selectable Grad-CAM, deterministic diagnostic inputs, and an explicitly labeled offline demonstration mode.

Fixed brittle hard-coded layer lists, deprecated `pretrained=True` use in the legacy loader, zero-division normalization of constant filters, missing exception-safe hook removal, output snapshots vulnerable to in-place mutation, and working-directory-dependent labels in the new app. Added weights-matched preprocessing, input validation, temporary bounded sessions, serialized model access, same-origin/host restrictions, and scientific interpretation guidance.

Added unit/API tests, a Chromium interaction smoke test, CI configuration, a non-root CPU Dockerfile, setup/deployment documentation, and an explicit validation record. Preserved the original notebook helper names and the legacy requirements filename alias. The UI command changed from `streamlit run interface.py` to `python interface.py`.
