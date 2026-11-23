"""Generate a small, reproducible image-augmentation dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.augment import apply_transform


TRANSFORMS = ("dark", "bright", "low_contrast", "blur", "glare")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def generate(input_dir: Path, output_dir: Path, variants: int) -> tuple[int, int]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be the same as or inside the input directory")
    if not 1 <= variants <= len(TRANSFORMS):
        raise ValueError(f"variants must be between 1 and {len(TRANSFORMS)}")

    sources = image_files(input_dir)
    if not sources:
        raise ValueError(f"No supported images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fields = ("source", "output", "transform", "parameters", "width", "height")
    generated = 0
    failures = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()

        for source in sources:
            relative_source = source.relative_to(input_dir)
            try:
                with Image.open(source) as opened:
                    original = opened.convert("RGB")
            except (OSError, UnidentifiedImageError) as error:
                print(f"Skipped unreadable image {relative_source}: {error}")
                failures += 1
                continue

            width, height = original.size
            for transform in TRANSFORMS[:variants]:
                augmented, parameters = apply_transform(original, transform)
                relative_output = relative_source.with_name(
                    f"{relative_source.stem}__{transform}.jpg"
                )
                destination = output_dir / relative_output
                destination.parent.mkdir(parents=True, exist_ok=True)
                augmented.save(destination, format="JPEG", quality=95, optimize=True)
                writer.writerow(
                    {
                        "source": relative_source.as_posix(),
                        "output": relative_output.as_posix(),
                        "transform": transform,
                        "parameters": parameters,
                        "width": width,
                        "height": height,
                    }
                )
                generated += 1

    print(f"Source images: {len(sources)}; generated images: {generated}; skipped: {failures}")
    print(f"Manifest: {manifest_path}")
    return generated, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Source image folder")
    parser.add_argument("--output", type=Path, default=Path("data/generated"), help="Output folder")
    parser.add_argument(
        "--variants", type=int, default=len(TRANSFORMS),
        help=f"Number of fixed transformations per image (1-{len(TRANSFORMS)})",
    )
    args = parser.parse_args()
    _, failures = generate(args.input, args.output, args.variants)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
