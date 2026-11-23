"""Small, deterministic image transformations for dataset augmentation."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def adjust_brightness(pixels: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel intensity; values below 1 darken and values above 1 brighten."""
    if factor < 0:
        raise ValueError("factor must be non-negative")
    adjusted = pixels.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(pixels: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast around the image's mean intensity."""
    if factor < 0:
        raise ValueError("factor must be non-negative")
    values = pixels.astype(np.float32)
    mean = values.mean(axis=(0, 1), keepdims=True)
    adjusted = (values - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply a soft defocus-like blur while preserving image dimensions."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def simulate_glare(image: Image.Image, opacity: int = 115) -> Image.Image:
    """Add a soft elliptical light reflection in the upper-right image area."""
    if not 0 <= opacity <= 255:
        raise ValueError("opacity must be between 0 and 255")

    image = image.convert("RGB")
    width, height = image.size
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    box = (
        int(width * 0.48),
        int(height * 0.03),
        int(width * 1.02),
        int(height * 0.52),
    )
    draw.ellipse(box, fill=opacity)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(width, height) * 0.04)))
    light = Image.new("RGB", image.size, (255, 255, 245))
    return Image.composite(light, image, mask)


def apply_transform(image: Image.Image, transform: str) -> tuple[Image.Image, str]:
    """Apply one named transformation and return the image and parameter summary."""
    image = image.convert("RGB")

    if transform == "dark":
        pixels = adjust_brightness(np.asarray(image), 0.45)
        return Image.fromarray(pixels), "factor=0.45"
    if transform == "bright":
        pixels = adjust_brightness(np.asarray(image), 1.35)
        return Image.fromarray(pixels), "factor=1.35"
    if transform == "low_contrast":
        pixels = adjust_contrast(np.asarray(image), 0.55)
        return Image.fromarray(pixels), "factor=0.55"
    if transform == "blur":
        return gaussian_blur(image, 1.6), "radius=1.6"
    if transform == "glare":
        return simulate_glare(image), "opacity=115;position=upper-right"

    raise ValueError(f"Unknown transformation: {transform}")
