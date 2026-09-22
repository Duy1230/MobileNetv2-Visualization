"""Real Chromium smoke test. No trained-weight download is needed.

Run: python tests/browser_smoke.py
Install a browser first: python -m playwright install chromium
Set CHROMIUM_PATH to use an existing Chromium executable.
"""
import base64
import json
import io
import urllib.error
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.getenv("MNV2_TEST_OUTPUT", str(ROOT / "test-artifacts")))
OUTPUT.mkdir(parents=True, exist_ok=True)
PORT = os.getenv("MNV2_TEST_PORT", "8781")
BASE = f"http://127.0.0.1:{PORT}"
env = dict(os.environ, MNV2_WEIGHTS="untrained", PORT=PORT, MNV2_HOST="127.0.0.1",
           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MNV2_NUM_THREADS="2")
log = open(OUTPUT / "server.log", "w")
process = subprocess.Popen([sys.executable, "interface.py"], cwd=ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(100):
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            if process.poll() is not None:
                raise RuntimeError("Server exited; inspect test-artifacts/server.log")
            time.sleep(.1)
    else:
        raise TimeoutError("Server did not start")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,
            executable_path=os.getenv("CHROMIUM_PATH"), args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        if os.getenv("MNV2_BROWSER_BRIDGE") == "1":
            # Restricted sandboxes can render local HTML but prohibit navigation.
            # Keep browser policy intact; bridge fetch to the same live local API.
            def transport(payload):
                request = urllib.request.Request(BASE + payload["url"],
                    data=base64.b64decode(payload["body"]) if payload.get("body") else None,
                    method=payload["method"], headers=payload.get("headers", {}))
                try:
                    response = urllib.request.urlopen(request, timeout=30)
                except urllib.error.HTTPError as exception:
                    response = exception
                with response:
                    return {"status":response.status, "headers":dict(response.headers),
                            "body":base64.b64encode(response.read()).decode()}
            page.expose_function("__transport", transport)
            html = (ROOT / "static/index.html").read_text()
            html = html.replace('<link rel="stylesheet" href="/app.css">',
                                '<style>' + (ROOT / "static/app.css").read_text() + '</style>')
            html = html.replace('<script src="/app.js" defer></script>', '')
            page.set_content(html)
            page.evaluate("""() => {
                window.fetch = async (url, options = {}) => {
                    let body = '';
                    if (options.body) {
                        const bytes = new Uint8Array(await new Response(options.body).arrayBuffer());
                        for (const byte of bytes) body += String.fromCharCode(byte);
                    }
                    const result = await window.__transport({url:String(url),
                        method:options.method || 'GET', headers:options.headers || {}, body:btoa(body)});
                    return new Response(result.status === 204 ? null :
                        Uint8Array.from(atob(result.body), x => x.charCodeAt(0)),
                        {status:result.status, headers:result.headers});
                };
            }""")
            page.add_script_tag(content=(ROOT / "static/app.js").read_text())
        else:
            page.goto(BASE)
        page.wait_for_selector("#activation-grid .channel-card", timeout=30000)
        assert page.locator("#activation-grid .channel-card").count() == 32
        assert "UNTRAINED" in page.locator("#demo-warning").inner_text()
        assert page.locator("#metric-layers").inner_text() == "52"
        page.screenshot(path=str(OUTPUT / "desktop.png"), full_page=True)
        page.locator("#layer-select").select_option("features.2.conv.1.0")
        # Locator assertions retry without eval(), preserving the app's strict CSP.
        expect(page.locator("#activation-shape")).to_have_text("96 × 56 × 56", timeout=30000)
        page.locator("#next-page").click()
        expect(page.locator("#page-label")).to_have_text("2 / 3", timeout=30000)
        page.locator("#sort").select_option("index")
        expect(page.locator("#page-label")).to_have_text("1 / 3", timeout=30000)
        page.locator("#activation-grid .channel-card").first.click()
        assert page.locator("#channel-dialog").is_visible()
        assert "C000" in page.locator("#dialog-title").inner_text()
        page.keyboard.press("Escape")
        assert not page.locator("#channel-dialog").is_visible()
        page.locator("#tab-filters").click()
        page.locator("#layer-select").select_option("features.18.0")
        page.wait_for_selector("#mixing-view", state="visible")
        assert "1280 output × 320 input" in page.locator("#mixing-caption").inner_text()
        page.evaluate("() => { document.activeElement.blur(); window.scrollTo(0, 0); }")
        page.screenshot(path=str(OUTPUT / "kernels.png"), full_page=True)
        page.locator("#tab-explain").click()
        page.locator("#class-target").fill("0 · tench")
        page.locator("#cam-button").click()
        page.wait_for_selector("#cam-results", state="visible", timeout=30000)
        assert "tench" in page.locator("#cam-caption").inner_text()
        page.evaluate("() => { document.activeElement.blur(); window.scrollTo(0, 0); }")
        page.screenshot(path=str(OUTPUT / "explain.png"), full_page=True)
        with page.expect_download() as downloaded:
            page.locator("#cam-download").click()
        assert downloaded.value.suggested_filename == "gradcam-overlay.png"
        page.locator("#tab-architecture").click()
        assert page.locator("#architecture-body tr").count() == 52
        page.locator("#architecture-body [data-layer='features.0.0']").click()
        page.wait_for_selector("#activation-grid .channel-card")
        with page.expect_download(timeout=30000) as downloaded:
            page.locator("#export-button").click()
        assert downloaded.value.suggested_filename == "mobilenet-analysis.zip"
        # Tabs support standard left/right keyboard navigation.
        page.locator("#tab-activations").focus()
        page.keyboard.press("ArrowRight")
        assert page.locator("#tab-filters").get_attribute("aria-selected") == "true"
        page.keyboard.press("ArrowLeft")
        assert page.locator("#tab-activations").get_attribute("aria-selected") == "true"
        for width in [1440, 768, 390]:
            page.set_viewport_size({"width":width,"height":844})
            page.wait_for_timeout(150)
            assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), f"Overflow at {width}px"
        page.evaluate("() => window.scrollTo(0, 0)")
        page.screenshot(path=str(OUTPUT / "mobile.png"), full_page=True)
        page.locator("#menu-button").click()
        assert page.locator("#menu-button").get_attribute("aria-expanded") == "true"
        upload = io.BytesIO()
        Image.new("RGB", (360, 240), (80, 140, 100)).save(upload, format="PNG")
        page.locator("#image-upload").set_input_files({"name":"diagnostic.png",
            "mimeType":"image/png", "buffer":upload.getvalue()})
        expect(page.locator("#input-caption")).to_contain_text("Uploaded image", timeout=30000)
        page.wait_for_selector("#activation-grid .channel-card")
        page.locator("#menu-button").click()
        page.locator("#clear-session").click()
        assert page.locator("#prediction-status").inner_text() == "Session cleared"
        assert page.locator("#export-button").is_disabled()
        assert not errors, errors
        browser.close()
    print(json.dumps({"result":"passed", "browser":"Chromium", "viewports":[1440,768,390],
                      "javascript_errors":errors, "transport":"bridge" if os.getenv("MNV2_BROWSER_BRIDGE") == "1" else "HTTP", "screenshots":str(OUTPUT)}, indent=2))
finally:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()
