"""Backwards-compatible helpers for notebooks using the original project."""
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from engine import capture_outputs, get_layer, read_image

__all__ = ["read_image", "show_image", "load_mobileNet", "normalize_filter",
           "get_layer_by_name", "preprocess_image", "get_multiple_intermediate_outputs"]


def load_mobileNet():
    return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()


def preprocess_image(image):
    # The legacy helper retains the original V1 model + 256px resize pairing.
    return MobileNet_V2_Weights.IMAGENET1K_V1.transforms()(image).unsqueeze(0)


def normalize_filter(filters):
    values = filters.detach().clone()
    span = values.max() - values.min()
    return torch.zeros_like(values) if span < 1e-12 else (values - values.min()) / span


def get_layer_by_name(model, layer_name):
    return get_layer(model, layer_name)


def get_multiple_intermediate_outputs(model, input_tensor, layer_names):
    # A caller sharing this model across threads must provide its own lock.
    return capture_outputs(model, input_tensor, layer_names)


def show_image(image, color_map="gray"):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=color_map)
    plt.axis("off")
    plt.show()
