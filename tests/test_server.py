import io
import json
import zipfile

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
from engine import Engine, png_bytes, sample_image


@pytest.fixture(scope="module")
def client():
    torch.set_num_threads(2)
    server.engines["untrained"] = Engine("untrained")
    with TestClient(server.app) as connection:
        yield connection


@pytest.fixture
def analysis(client):
    response = client.post("/api/analyze?sample=shapes&weights=untrained")
    assert response.status_code == 200
    data = response.json()
    yield data
    client.delete(f"/api/analysis/{data['id']}")


def test_health_and_static(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/classes").json()) == 1000
    page = client.get("/")
    assert page.status_code == 200 and "Vision Lab" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/app.js").status_code == 200
    assert client.get("/app.css").status_code == 200
    assert client.get("/api/health").headers["cache-control"] == "no-store"


def test_trained_loading_failure_is_not_silent_fallback(client, monkeypatch):
    server.engines.pop("IMAGENET1K_V2", None)
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(server, "Engine", fail)
    response = client.post("/api/analyze?sample=shapes&weights=IMAGENET1K_V2")
    assert response.status_code == 503
    assert "Offline demo" in response.json()["detail"]
    assert "IMAGENET1K_V2" not in server.engines


def test_summary(analysis):
    assert not analysis["model"]["trained"]
    assert "UNTRAINED" in analysis["warning"]
    assert len(analysis["predictions"]) == 5
    assert analysis["crop"].startswith("data:image/png;base64,")
    assert analysis["model"]["preprocessing"]["crop"] == [224]


def test_upload(client):
    response = client.post("/api/analyze?weights=untrained",
                           content=png_bytes(sample_image("spectrum")))
    assert response.status_code == 200 and response.json()["source"] == "Uploaded image"
    client.delete(f"/api/analysis/{response.json()['id']}")


@pytest.mark.parametrize("path,body,status", [
    ("/api/analyze?weights=untrained", b"broken", 422),
    ("/api/analyze?weights=untrained", b"", 422),
    ("/api/analyze?sample=unknown&weights=untrained", b"", 422),
    ("/api/analyze?sample=shapes&weights=unknown", b"", 422),
    ("/api/analyze?sample=shapes&weights=untrained", b"body", 422),
])
def test_invalid_ingestion(client, path, body, status):
    assert client.post(path, content=body).status_code == status


def test_upload_limit(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 20)
    assert client.post("/api/analyze", content=b"x" * 21).status_code == 413


def test_activations_and_weights(client, analysis):
    prefix = f"/api/analysis/{analysis['id']}"
    data = client.get(prefix + "/activations").json()
    assert data["shape"] == [32, 112, 112] and len(data["channels"]) == 32
    data = client.get(prefix + "/filters?layer=features.18.0").json()
    assert data["matrix_shape"] == [1280, 320] and data["input_channels"] == 320
    assert client.get(prefix + "/activations?layer=missing").status_code == 422
    assert client.get(prefix + "/activations?page=10000").status_code == 422
    assert client.get(prefix + "/activations?per_page=1000").status_code == 422
    assert client.get(prefix + "/filters?input_channel=10000").status_code == 422


def test_cam(client, analysis):
    prefix = f"/api/analysis/{analysis['id']}"
    result = client.post(prefix + "/cam?target=0").json()
    assert result["class_index"] == 0 and not result["trained"]
    assert result["overlay"].startswith("data:image/png;base64,")
    assert client.post(prefix + "/cam?target=1000").status_code == 422
    assert client.post(prefix + "/cam?target=-1").status_code == 422
    assert client.post(prefix + "/cam?target=0&alpha=2").status_code == 422


def test_export(client, analysis):
    response = client.get(f"/api/analysis/{analysis['id']}/export")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"report.json", "activations.npy", "predictions.csv", "model-input.png"}
        report = json.loads(archive.read("report.json"))
        assert not report["model"]["trained"]
        assert report["export"]["activation_shape"] == [32, 112, 112]
        assert "crop" not in report and report["id"] == "omitted"
        values = np.load(io.BytesIO(archive.read("activations.npy")), allow_pickle=False)
        assert values.shape == (32, 112, 112) and values.dtype == np.float32
        assert len(archive.read("predictions.csv").decode().splitlines()) == 1001


def test_session_deletion(client, analysis):
    prefix = f"/api/analysis/{analysis['id']}"
    assert client.delete(prefix).status_code == 204
    assert client.get(prefix + "/activations").status_code == 404
    assert client.get("/api/analysis/not-a-real-token/activations").status_code == 404


def test_store_lru_and_expiry(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: clock[0])
    store = server.SessionStore(capacity=2, ttl=10)
    first, second = store.put("first"), store.put("second")
    assert store.get(first) == "first"
    third = store.put("third")
    with pytest.raises(HTTPException): store.get(second)
    assert store.get(third) == "third"
    clock[0] = 11
    with pytest.raises(HTTPException): store.get(first)
    assert not store.items


def test_cross_origin_and_host_protection(client):
    response = client.post("/api/analyze?sample=shapes&weights=untrained",
                           headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert client.get("/api/health", headers={"Host":"evil.example"}).status_code == 400
