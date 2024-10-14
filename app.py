import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image


def read_image(path):
    image = Image.open(path).convert('RGB')
    return image


def show_image(image, color_map="gray"):
    plt.imshow(image, cmap=color_map)
    plt.axis("off")
    plt.show()


def load_mobileNet():
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.eval()
    return model


def normalize_filter(filter):
    min_val = filter.min()
    max_val = filter.max()
    normalized_filter = (filter - min_val) / (max_val - min_val)
    return normalized_filter


def get_layer_by_name(model, layer_name):
    return model.get_submodule(layer_name)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    img_tensor = transform(image)

    # Normalize the image
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)

    return img_tensor.unsqueeze(0)  # Add batch dimension


def get_multiple_intermediate_outputs(model, input_tensor, layer_names):
    intermediate_outputs = {}

    def hook_fn(name):
        def hook(module, input, output):
            intermediate_outputs[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        pred = model(input_tensor)

    for hook in hooks:
        hook.remove()

    return intermediate_outputs, pred
