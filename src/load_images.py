from pathlib import Path

import numpy as np
from PIL import Image

RAW_DIR = Path("data/raw")


def load_image(path):
    """Open an image and return it as a NumPy array of shape (height, width, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def main():
    images = sorted(RAW_DIR.glob("*.jpg"))
    if not images:
        print("No images found in data/raw")
        return

    first = images[0]
    pixels = load_image(first)

    print(f"File: {first.name}")
    print(f"Shape (height, width, channels): {pixels.shape}")
    print(f"Average brightness (0 to 255): {pixels.mean():.1f}")


if __name__ == "__main__":
    main()