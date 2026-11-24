import cv2
import numpy as np


def adjust_brightness(pixels, factor):
    """Multiply pixel values by factor. factor < 1 darkens, factor > 1 brightens."""
    out = pixels.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(pixels, kernel_size=15):
    """Blur the image to simulate motion. Larger kernel_size gives stronger blur.
    kernel_size must be an odd number."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(pixels, (kernel_size, kernel_size), 0)

def add_glare(pixels, center_x=0.7, center_y=0.3, strength=150, radius=0.15):
    """Add a soft bright spot to simulate glare.
    center_x and center_y are fractions of width and height (0 to 1).
    strength is how much brightness to add at the centre of the spot."""
    h, w = pixels.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = center_x * w, center_y * h
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    sigma = radius * max(h, w)
    mask = np.exp(-dist2 / (2 * sigma ** 2))
    out = pixels.astype(np.float32) + strength * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)