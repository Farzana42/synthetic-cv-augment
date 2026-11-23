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