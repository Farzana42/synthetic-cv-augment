from pathlib import Path

from PIL import Image

from src.load_images import load_image
from src.augment import adjust_brightness

original = load_image("data/raw/20260717_174101.jpg")
darker = adjust_brightness(original, 0.4)

Path("data/output").mkdir(parents=True, exist_ok=True)
Image.fromarray(darker).save("data/output/dark_test.jpg")
print("Saved data/output/dark_test.jpg")

from src.augment import add_blur

blurred = add_blur(original, 25)
Image.fromarray(blurred).save("data/output/blur_test.jpg")
print("Saved data/output/blur_test.jpg")

from src.augment import add_glare

glare = add_glare(original)
Image.fromarray(glare).save("data/output/glare_test.jpg")
print("Saved data/output/glare_test.jpg")