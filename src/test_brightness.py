from pathlib import Path

from PIL import Image

from src.load_images import load_image
from src.augment import adjust_brightness

original = load_image("data/raw/20260717_174101.jpg")
darker = adjust_brightness(original, 0.4)

Path("data/output").mkdir(parents=True, exist_ok=True)
Image.fromarray(darker).save("data/output/dark_test.jpg")
print("Saved data/output/dark_test.jpg")