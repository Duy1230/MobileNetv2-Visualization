import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app import normalize_filter, preprocess_image
from engine import (MAX_BYTES, Engine, capture_outputs, channel_page, colorize,
                    normalize, png_bytes, read_image, sample_image)


@pytest.fixture(scope="session")
def engine():
    torch.set_num_threads(2)
    return Engine("untrained")


@pytest.fixture
def tensor(engine):
    return engine.prepare(sample_image("shapes"))[0]


def hook_count(model):
    return sum(len(module._forward_hooks) for module in model.modules())


@pytest.mark.parametrize("name", ["shapes", "stripes", "spectrum"])
def test_samples(name):
    image = sample_image(name)
    assert image.mode == "RGB" and image.size == (512, 512)
    assert png_bytes(image) == png_bytes(sample_image(name))


@pytest.mark.parametrize("raw", [b"", b"not an image", b"x" * (MAX_BYTES + 1)])
def test_invalid_image(raw):
    with pytest.raises(ValueError):
        read_image(raw)


def test_transparent_image_is_composited():
    image = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
    decoded = read_image(png_bytes(image))
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_grayscale_and_stream():
    image = read_image(io.BytesIO(png_bytes(Image.new("L", (8, 9), 120))))
    assert image.mode == "RGB" and image.size == (8, 9)


def test_exif_orientation():
    image = Image.new("RGB", (12, 24), "red")
    exif = image.getexif()
    exif[274] = 6
    buf = io.BytesIO()
    image.save(buf, "JPEG", exif=exif)
    assert read_image(buf.getvalue()).size == (24, 12)


def test_pixels_rejected_before_decode(monkeypatch):
    monkeypatch.setattr("engine.MAX_PIXELS", 30)
    with pytest.raises(ValueError, match="megapixel"):
        read_image(png_bytes(Image.new("RGB", (8, 8))))


def test_animation_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "PNG", save_all=True,
        append_images=[Image.new("RGB", (8, 8), "blue")], duration=100)
    with pytest.raises(ValueError, match="still"):
        read_image(buf.getvalue())


def test_normalization():
    assert not normalize(np.ones((3, 3))).any()
    assert np.allclose(normalize(np.array([-2, 0, 2])), [0, .5, 1])
    assert not normalize_filter(torch.ones(2, 2)).any()
    assert torch.isfinite(normalize_filter(torch.tensor([-1., 2.]))).all()


def test_model_profile(engine):
    info = engine.metadata()
    assert info["parameters"] == 3_504_872
    assert len(info["layers"]) == 52 and len(info["stages"]) == 19
    assert info["macs"] == 300_774_272
    assert {layer["kind"] for layer in info["layers"]} == {"spatial", "pointwise", "depthwise"}
    assert not engine.trained
    assert info["preprocessing"]["resize"] == [232]
    assert hook_count(engine.model) == 0


def test_exact_preprocessing(engine):
    image = sample_image("shapes")
    tensor, crop = engine.prepare(image)
    assert tensor.shape == (1, 3, 224, 224)
    assert torch.equal(tensor[0], engine.preset.transforms()(image))
    assert crop.size == (224, 224)
    assert preprocess_image(image).shape == tensor.shape


def test_prediction(engine, tensor):
    probabilities, elapsed = engine.predict(tensor)
    assert probabilities.shape == (1000,)
    assert probabilities.sum().item() == pytest.approx(1, abs=1e-6)
    assert elapsed > 0 and torch.isfinite(probabilities).all()
    assert hook_count(engine.model) == 0


def test_capture_survives_inplace_operations():
    model = nn.Sequential(nn.Identity(), nn.ReLU(inplace=True))
    result, _ = capture_outputs(model, torch.tensor([-3., 2.]), ["0"])
    assert result["0"].tolist() == [-3., 2.]
    assert hook_count(model) == 0


def test_capture_cleanup_on_failure():
    class Fails(nn.Module):
        def forward(self, value):
            raise RuntimeError("intentional failure")
    model = nn.Sequential(nn.Identity(), Fails())
    with pytest.raises(RuntimeError):
        capture_outputs(model, torch.ones(1), ["0"])
    assert hook_count(model) == 0
    with pytest.raises(ValueError):
        capture_outputs(model, torch.ones(1), ["0", "missing"])
    assert hook_count(model) == 0


def test_invalid_layer(engine, tensor):
    with pytest.raises(ValueError):
        engine.activation(tensor, "classifier.1")


@pytest.mark.parametrize("name,shape", [
    ("features.0.0", (32, 112, 112)),
    ("features.1.conv.0.0", (32, 112, 112)),
    ("features.18.0", (1280, 7, 7)),
])
def test_activations(engine, tensor, name, shape):
    values = engine.activation(tensor, name)
    assert values.shape == shape and np.isfinite(values).all()
    assert hook_count(engine.model) == 0


def test_kernel_types(engine):
    assert engine.filters("features.0.0", 2).shape == (32, 3, 3)
    assert engine.filters("features.1.conv.0.0").shape == (32, 3, 3)
    assert engine.filters("features.18.0").shape == (1280, 1, 1)
    assert engine.mixing_matrix("features.18.0").shape == (1280, 320)
    with pytest.raises(ValueError):
        engine.filters("features.1.conv.0.0", 1)


def test_channel_page():
    values = np.arange(10 * 4 * 4, dtype=np.float32).reshape(10, 4, 4)
    page = channel_page(values, per_page=4)
    assert [c["index"] for c in page["channels"]] == [9, 8, 7, 6]
    assert page["pages"] == 3 and page["range"] == [0, 159]
    assert page["channels"][0]["image"].startswith("data:image/png;base64,")
    assert len(channel_page(values, page=2, per_page=4)["channels"]) == 2
    assert [c["index"] for c in channel_page(values, sort="index")["channels"]] == list(range(10))
    assert colorize(np.zeros((4, 4)), "gray").size == (4, 4)


@pytest.mark.parametrize("kwargs", [{"page":-1}, {"page":100}, {"per_page":0},
                                    {"per_page":65}, {"sort":"oops"},
                                    {"normalization":"oops"}, {"palette":"oops"}])
def test_invalid_display_options(kwargs):
    with pytest.raises(ValueError):
        channel_page(np.zeros((2, 2, 2)), **kwargs)


def test_gradcam(engine, tensor):
    heat = engine.gradcam(tensor, "features.18.0", 0)
    assert heat.shape == (224, 224) and np.isfinite(heat).all()
    assert 0 <= heat.min() <= heat.max() <= 1
    assert hook_count(engine.model) == 0
    assert all(p.grad is None for p in engine.model.parameters())
    with pytest.raises(ValueError):
        engine.gradcam(tensor, "features.18.0", 1000)


def test_gradcam_against_analytic_reference(engine):
    """A one-channel linear model has an analytically known Grad-CAM."""
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1, bias=False)
            self.conv.weight.data.fill_(1)
        def forward(self, x):
            return self.conv(x).mean((2, 3))
    toy = Engine.__new__(Engine)
    import threading
    toy.lock = threading.RLock()
    toy.model = Toy().eval().requires_grad_(False)
    toy.device = torch.device("cpu")
    toy.categories = ["positive"]
    toy.layer_names = {"conv"}
    x = torch.arange(224, dtype=torch.float32).repeat(224, 1)[None, None].repeat(1, 3, 1, 1)
    heat = toy.gradcam(x, "conv", 0)
    expected = np.tile(np.linspace(0, 1, 224), (224, 1))
    assert np.allclose(heat, expected, atol=1e-5)
    assert hook_count(toy.model) == 0


def test_concurrent_inference_is_isolated(engine, tensor):
    names = ["features.0.0", "features.18.0"] * 3
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda name: engine.activation(tensor, name), names))
    for name, output in zip(names, outputs):
        assert output.shape[0] == (32 if name == "features.0.0" else 1280)
    assert hook_count(engine.model) == 0
