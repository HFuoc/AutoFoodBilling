# Tray Cell Training

This is the recommended path when the goal is:

```text
tray image -> detect each tray cell -> crop each cell -> classify food in that cell
```

The current `food_item` detector pipeline can crop visible food regions, but it does not know about empty cells or fixed tray compartments. For tray-based billing, train a separate one-class detector named `tray_cell`.

## 1. Prepare Real Tray Images

Put top-down tray photos here:

```text
data/raw/tray_cells/
```

Use the same camera angle and distance as the demo whenever possible.

## 2. Annotate Tray Cells

Create:

```text
data/annotations/tray_cells.json
```

An example is available at:

```text
data/annotations/tray_cells.example.json
```

Format:

```json
{
  "images": [
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
  ]
}
```

`box` uses pixel coordinates `[x1, y1, x2, y2]`. The `label` field is metadata for review; detector is trained with one class, `tray_cell`.

### Fast Path For The Single Competition Tray

If the competition uses only the current stainless tray shape, seed all five cell boxes from the fixed template instead of relying on detector suggestions:

```powershell
.\.venv\Scripts\python.exe ml\training\seed_fixed_tray_annotations.py --image-dir data\raw\tray_with_food --output data\annotations\tray_cells_fixed_template.json
```

Open:

```text
data/generated/annotation_previews_fixed_template/
```

Then fill the `label` for each cell in `data/annotations/tray_cells_fixed_template.json`, set reviewed records to `true`, and use that file for crop extraction:

```powershell
.\.venv\Scripts\python.exe ml\training\extract_tray_cell_crops.py --annotations data\annotations\tray_cells_fixed_template.json --output data\processed\tray_cell_food_classes_fixed --overwrite
.\.venv\Scripts\python.exe ml\training\prepare_supplemented_cnn_dataset.py --primary data\processed\tray_cell_food_classes_fixed --supplement data\raw\food_classes --output data\processed\tray_cell_food_classes_supplemented --max-supplement-per-class 250 --overwrite
.\.venv\Scripts\python.exe ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes_supplemented --classes data\processed\tray_cell_food_classes_supplemented\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16 --image-size 160
```

`prepare_supplemented_cnn_dataset.py` copies every labeled tray-cell crop first, then adds a capped sample from the single-dish folders in `data/raw/food_classes`. It follows `configs/classes.json` for supplement images, so the `Khay` folder is intentionally ignored. Classes that exist only in tray-cell crops, such as `empty`, are preserved.

For inference on the same tray, the CLI now defaults to the fixed template crop:

```powershell
.\.venv\Scripts\python.exe ml\inference\run_cell_pipeline.py data\raw\tray_with_food\sample.jpg
```

If your raw data is split into empty trays and trays with food, seed annotations from:

```text
data/raw/empty_trays/
data/raw/tray_with_food/
```

Generate detector suggestions and preview images:

```powershell
python ml\training\seed_tray_cell_annotations.py --output data\annotations\tray_cells_auto.json
```

Open:

```text
data/generated/annotation_previews_auto/
```

Only train from boxes that have been visually checked. For records you approve in
`data/annotations/tray_cells_auto.json`, set:

```json
"reviewed": true
```

For a quick experiment only, you can accept the detector suggestions directly:

```powershell
python ml\training\seed_tray_cell_annotations.py --output data\annotations\tray_cells_auto.json --accept-suggestions
```

That is faster, but if boxes are wrong it will teach detector the wrong crop.

## 3. Preview And Validate Annotations

Draw the annotated boxes before training:

```powershell
python ml\training\preview_tray_cell_annotations.py --annotations data\annotations\tray_cells.json
```

Preview images are written to:

```text
data/generated/annotation_previews/
```

Check these images visually. If a box misses a compartment, fix `tray_cells.json` before generating training data.

Run the data checker after the visual pass:

```powershell
python ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json --skip-crops
```

The checker accepts `empty` by default. Use `--no-include-empty` if every label must be a menu item.

## 4. Optional Synthetic Bootstrap

If you have an annotated empty tray layout but not enough real filled trays yet, generate synthetic data:

```powershell
python ml\training\generate_synthetic_tray_cell_dataset.py --layouts data\annotations\tray_cells.json --samples 300 --overwrite
```

If you only have empty tray layouts and separate labeled dish photos, use the empty-layout annotation directly. The generator composites the already labeled food-class images from `data/raw/food_classes/` into annotated tray cells:

```powershell
python ml\training\generate_synthetic_tray_cell_dataset.py --layouts data\annotations\tray_layouts_empty.json --samples 1200 --empty-probability 0.15 --overwrite
```

This writes:

```text
data/generated/tray_cell_synthetic/detector/data.yaml
data/generated/tray_cell_synthetic/cnn/classes.json
data/generated/tray_cell_synthetic/metadata/synthetic_cells_val.json
```

You can bootstrap training from this synthetic dataset:

```powershell
python ml\training\train_detector.py --data data\generated\tray_cell_synthetic\detector\data.yaml --name tray_cell_synthetic --output ml\models\detector\cell_best.pt
python ml\training\train_cnn.py --data-root data\generated\tray_cell_synthetic\cnn --classes data\generated\tray_cell_synthetic\cnn\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16
python ml\inference\evaluate_cell_pipeline.py --annotations data\generated\tray_cell_synthetic\metadata\synthetic_cells_val.json --cell-detector-model ml\models\detector\cell_best.pt --cnn-model ml\models\cnn\cell_best.pt
```

Or run the whole synthetic bootstrap sequence:

```powershell
python ml\training\bootstrap_tray_cell_models.py --layouts data\annotations\tray_cells.json --samples 300 --overwrite
```

Use `--dry-run` to print the generated commands without training.

Synthetic data is useful for a first model, but real annotated tray photos should still be added for final accuracy.

## 5. Build detector Cell Dataset

```powershell
python ml\training\prepare_tray_cell_detector_dataset.py --overwrite
```

Output:

```text
data/processed/tray_cells_detector/data.yaml
```

## 6. Train Tray Cell Detector

```powershell
python ml\training\train_detector.py --data data\processed\tray_cells_detector\data.yaml --name tray_cell --output ml\models\detector\cell_best.pt
```

For the cleaned `empty_trays` + `tray_with_food` workflow, run the guarded retrain pipeline:

```powershell
python ml\training\train_tray_cell_detector.py --overwrite
```

By default this only trains from annotation records marked `reviewed=true`. After
you inspect the preview images, either mark good records as reviewed in
`data/annotations/tray_cells_auto.json`, or run a rough experiment with:

```powershell
python ml\training\train_tray_cell_detector.py --skip-seed --allow-unreviewed --overwrite
```

Use `--dry-run` to print the exact commands without training:

```powershell
python ml\training\train_tray_cell_detector.py --dry-run --overwrite
```

## 7. Extract Cell Crops For CNN

The same annotation can create a folder-per-class dataset of real cell crops:

```powershell
python ml\training\extract_tray_cell_crops.py --overwrite
```

Output:

```text
data/processed/tray_cell_food_classes/
```

Each labeled cell is saved under:

```text
data/processed/tray_cell_food_classes/<label>/
```

The extractor also writes a matching class config:

```text
data/processed/tray_cell_food_classes/classes.json
```

After extracting crops, run the checker again without `--skip-crops`:

```powershell
python ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json
```

## 8. Train Food Classifier

Train the CNN on the cell-crop dataset:

```powershell
python ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes --classes data\processed\tray_cell_food_classes\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16
```

For small datasets, prefer the default transfer-learning setup in `train_cnn.py`. It starts from ImageNet weights, uses class-balanced sampling, class weights, label smoothing, and tray-like augmentations. If you generated synthetic cell crops from the labeled dish photos, train the CNN on those crops:

```powershell
python ml\training\train_cnn.py --data-root data\generated\tray_cell_synthetic\cnn --classes data\generated\tray_cell_synthetic\cnn\classes.json --output ml\models\cnn\cell_best.pt --epochs 40 --batch-size 16 --lr 0.0002
```

If you also have real tray-cell crops, merge real and synthetic crops before training:

```powershell
python ml\training\merge_cnn_datasets.py --inputs data\processed\tray_cell_food_classes data\generated\tray_cell_synthetic\cnn --output data\processed\tray_cell_food_classes_combined --overwrite
python ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes_combined --classes data\processed\tray_cell_food_classes_combined\classes.json --output ml\models\cnn\cell_best.pt --epochs 40 --batch-size 16 --lr 0.0002
```

The original classifier trained on `data/raw/food_classes/` can still be used for early testing, but real tray-cell crops usually match the final camera setup better.

If cells can be empty, label those cells as `empty`; the generated `classes.json` will include it automatically. You can also copy `configs/classes_with_empty.example.json` if you want a fixed class order before enough cell crops exist for every class.

```powershell
python ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes --classes configs\classes_with_empty.example.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16
```

## 9. Run Cell Pipeline

```powershell
python ml\inference\run_cell_pipeline.py data\raw\real_trays_test\sample.jpg
```

The script detects tray cells, saves cell crops under `data/generated/cell_crops/`, classifies each crop, and builds the bill from non-empty cells.

## 10. Evaluate Trained Models

Evaluate the detector against annotated tray cells:

```powershell
python ml\inference\evaluate_cell_pipeline.py --annotations data\annotations\tray_cells.json --skip-classifier
```

Evaluate both detector and classifier:

```powershell
python ml\inference\evaluate_cell_pipeline.py --annotations data\annotations\tray_cells.json --cell-detector-model ml\models\detector\cell_best.pt --cnn-model ml\models\cnn\cell_best.pt
```

The report includes cell detection recall, precision, mean IoU, and label accuracy when the classifier is enabled.
