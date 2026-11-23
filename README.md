# Synthetic CV Augment

A small Python project for generating controlled image variations and checking that an image dataset is complete and readable. It demonstrates an image-data workflow relevant to computer-vision products: load source images, apply repeatable visual conditions, record each transformation, and validate the generated files.

## What it does

- Creates darker, brighter, lower-contrast, blurred, and glare-affected versions of source images.
- Writes a CSV manifest with the source file, generated file, transformation, and parameters.
- Checks that every manifest output exists, opens as an image, and retains the source dimensions.
- Uses a fixed set of transformations so that results can be reproduced and compared.

The transformations change image appearance but do not create new objects or ground-truth annotations. If source images are organized in class subfolders, the folder structure is retained. This project validates dataset files and manifest consistency; it does not measure a computer-vision model's accuracy.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Generate image variations

Place source images under `data/raw/`. You can also use a different input folder.

```bash
python -m src.generate_dataset --input data/raw --output data/generated --variants 5
```

The command creates one output for each transformation per source image and writes `manifest.csv` in the output folder. Existing files with the same generated names are replaced; unrelated files are left untouched.

## Validate the generated dataset

```bash
python -m src.validate_dataset --dataset data/generated
```

The validator reports the number of manifest entries checked and any missing, unreadable, duplicated, or dimension-mismatched files. It returns a non-zero exit status if it finds an issue.

## Project structure

```text
src/
  augment.py          Image transformations
  generate_dataset.py Dataset generation and manifest writing
  validate_dataset.py Dataset integrity checks
data/
  raw/                Source images
  generated/          Generated locally; ignored by Git
```

## Scope and next steps

This is a learning project and a starting point for synthetic data work. Useful extensions include configurable transformation ranges, image annotations, a small visual review report, and evaluating how transformations affect a selected computer-vision model. Generated data should be reviewed for realism and label preservation before being used to train or evaluate a model.
