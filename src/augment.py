import numpy as np


def adjust_brightness(pixels, factor):
    """Multiply pixel values by factor. factor < 1 darkens, factor > 1 brightens."""
    out = pixels.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)