# Manual Accuracy Training

Use this when the frontend shows wrong boxes or wrong dish names.

## What To Correct

There are two separate models:

```text
detector cell detector: ml/models/detector/cell_best.pt
CNN food classifier: ml/models/cnn/cell_best.pt
```

- If boxes miss cells, overlap badly, or detect the wrong area: improve detector annotations.
- If boxes are okay but names are wrong: improve CNN labels/crops.

## Annotation File

Edit:

```text
data/annotations/tray_cells.json
```

Each tray image needs five cells:

```json
{
  "file": "data/raw/real_trays_test/screenshot_1781814961.png",
  "cells": [
    {
      "cell_id": "top_left",
      "box": [47, 44, 208, 165],
      "label": "Thịt kho"
    }
  ]
}
```

Labels must match `configs/menu.json`, or use `empty`.

Do not train on screenshots that already have red prediction boxes drawn on top. Use clean tray images whenever possible.

## Preview And Check

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe ml\training\preview_tray_cell_annotations.py --annotations data\annotations\tray_cells.json --output data\generated\annotation_previews_manual
.\.venv\Scripts\python.exe ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json --skip-crops --min-per-class 1
```

Open:

```text
data/generated/annotation_previews_manual/
```

Fix boxes and labels until the previews are correct.

## Recommended Manual Retrain Flow

Generate synthetic data from the clean empty-tray layout:

```powershell
.\.venv\Scripts\python.exe ml\training\generate_synthetic_tray_cell_dataset.py --layouts data\annotations\tray_layouts_empty.json --output data\generated\tray_cell_synthetic --samples 800 --val-ratio 0.2 --empty-probability 0.12 --seed 123 --overwrite
```

Prepare detector from your manually corrected tray annotations:

```powershell
.\.venv\Scripts\python.exe ml\training\prepare_tray_cell_detector_dataset.py --annotations data\annotations\tray_cells.json --output data\processed\tray_cells_detector_manual --overwrite
```

Merge synthetic detector and manual detector:

```powershell
.\.venv\Scripts\python.exe ml\training\merge_detector_datasets.py --inputs data\generated\tray_cell_synthetic\detector data\processed\tray_cells_detector_manual --output data\processed\tray_cells_detector_combined --overwrite
```

Train the cell detector:

```powershell
.\.venv\Scripts\python.exe ml\training\train_detector.py --data data\processed\tray_cells_detector_combined\data.yaml --base-model detector_nano.det --epochs 30 --image-size 416 --batch-size 4 --workers 0 --project ml\models\detector\runs --name tray_cell_manual --output ml\models\detector\cell_best.pt
```

Extract manually labeled real crops:

```powershell
.\.venv\Scripts\python.exe ml\training\extract_tray_cell_crops.py --annotations data\annotations\tray_cells.json --output data\processed\tray_cell_food_classes_manual --overwrite
```

Merge synthetic CNN crops and manual real crops:

```powershell
.\.venv\Scripts\python.exe ml\training\merge_cnn_datasets.py --inputs data\generated\tray_cell_synthetic\cnn data\processed\tray_cell_food_classes_manual --output data\processed\tray_cell_food_classes_combined --overwrite
```

Train the classifier:

```powershell
.\.venv\Scripts\python.exe ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes_combined --classes data\processed\tray_cell_food_classes_combined\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16 --image-size 160 --val-ratio 0.2 --num-workers 0
```

Test:

```powershell
.\.venv\Scripts\python.exe ml\inference\run_cell_pipeline.py data\raw\real_trays_test\screenshot_1781814961.png
```

## Practical Target

For a usable camera demo, add and annotate at least:

```text
10-20 clean tray images from the same camera
5 boxes per image
labels for each visible dish or empty
```

After every correction batch, retrain and test again.
