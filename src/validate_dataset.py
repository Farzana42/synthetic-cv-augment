"""Check generated image files against their CSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def validate(dataset_dir: Path) -> tuple[int, list[str]]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    issues: list[str] = []
    seen_outputs: set[str] = set()
    checked = 0

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"source", "output", "transform", "parameters", "width", "height"}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            checked += 1
            relative_output = Path(row["output"])
            output_path = (dataset_dir / relative_output).resolve()
            if dataset_dir not in output_path.parents:
                issues.append(f"Line {line_number}: output path escapes the dataset directory")
                continue

            normalized = relative_output.as_posix()
            if normalized in seen_outputs:
                issues.append(f"Line {line_number}: duplicate output path {normalized}")
            seen_outputs.add(normalized)

            if not output_path.is_file():
                issues.append(f"Line {line_number}: missing output {normalized}")
                continue

            try:
                with Image.open(output_path) as image:
                    image.verify()
                with Image.open(output_path) as image:
                    actual_size = image.size
                    actual_mode = image.mode
            except (OSError, UnidentifiedImageError) as error:
                issues.append(f"Line {line_number}: unreadable output {normalized}: {error}")
                continue

            try:
                expected_size = (int(row["width"]), int(row["height"]))
            except (TypeError, ValueError):
                issues.append(f"Line {line_number}: invalid dimensions in manifest")
                continue

            if actual_size != expected_size:
                issues.append(
                    f"Line {line_number}: dimensions for {normalized} are {actual_size}, "
                    f"expected {expected_size}"
                )
            if actual_mode != "RGB":
                issues.append(f"Line {line_number}: {normalized} has mode {actual_mode}, expected RGB")

    if checked == 0:
        issues.append("Manifest contains no generated image records")
    return checked, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/generated"), help="Dataset folder")
    args = parser.parse_args()

    checked, issues = validate(args.dataset)
    print(f"Manifest records checked: {checked}")
    if issues:
        print(f"Issues found: {len(issues)}")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("Dataset files and manifest are consistent.")


if __name__ == "__main__":
    main()
