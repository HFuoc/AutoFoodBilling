# Manual Tray Cell Training Checklist

Use this checklist when you want to train manually.

## 1. Add Real Tray Images

Put top-down tray images here:

```text
data/raw/tray_cells/
```

Use the same camera angle, distance, tray type, and lighting as the demo setup when possible.

If you only need a tray layout bootstrap, you can also reference existing tray images from:

```text
data/raw/empty_trays/
```

## 2. Fill The Annotation File

Edit:

```text
data/annotations/tray_cells.json
```

The file currently starts empty:

```json
{
  "images": []
}
```

For each tray image, add one entry like this:

```json
{
  "file": "data/raw/tray_cells/sample_001.jpg",
  "cells": [
    {
      "cell_id": "top_left",
      "box": [40, 60, 210, 230],
      "label": "Canh rau"
    },
    {
      "cell_id": "bottom_right",
      "box": [260, 250, 610, 520],
      "label": "Cơm trắng"
    }
  ]
}
```

`box` is `[x1, y1, x2, y2]` in pixels. Use one of the menu labels in `configs/menu.json`, or `empty` for an empty tray cell.

## 3. Preview And Check Annotations

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe ml\training\preview_tray_cell_annotations.py --annotations data\annotations\tray_cells.json
.\.venv\Scripts\python.exe ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json --skip-crops
```

Preview output:

```text
data/generated/annotation_previews/
```

Fix the boxes in `tray_cells.json` until the preview looks correct.

## 4. Prepare detector Cell Dataset

```powershell
.\.venv\Scripts\python.exe ml\training\prepare_tray_cell_detector_dataset.py --overwrite
```

Expected output:

```text
data/processed/tray_cells_detector/data.yaml
```

## 5. Train Tray Cell detector

```powershell
.\.venv\Scripts\python.exe ml\training\train_detector.py --data data\processed\tray_cells_detector\data.yaml --name tray_cell --output ml\models\detector\cell_best.pt
```

Expected model:

```text
ml/models/detector/cell_best.pt
```

## 6. Extract Real Cell Crops

```powershell
.\.venv\Scripts\python.exe ml\training\extract_tray_cell_crops.py --overwrite
.\.venv\Scripts\python.exe ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json
```

Expected output:

```text
data/processed/tray_cell_food_classes/
data/processed/tray_cell_food_classes/classes.json
```

## 7. Train Cell Crop Classifier

```powershell
.\.venv\Scripts\python.exe ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes --classes data\processed\tray_cell_food_classes\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16
```

Expected model:

```text
ml/models/cnn/cell_best.pt
```

## 8. Evaluate End To End

```powershell
.\.venv\Scripts\python.exe ml\inference\evaluate_cell_pipeline.py --annotations data\annotations\tray_cells.json --cell-detector-model ml\models\detector\cell_best.pt --cnn-model ml\models\cnn\cell_best.pt
```

## 9. Test A Real Tray Image

Put a demo image here:

```text
data/raw/real_trays_test/sample.jpg
```

Run:

```powershell
.\.venv\Scripts\python.exe ml\inference\run_cell_pipeline.py data\raw\real_trays_test\sample.jpg
```

The final output should list detected cells, predicted food labels, item prices, and the total bill.
