# Pipeline Bootstrap Status

This project now has a runnable bootstrap pipeline:

```text
tray image -> detect tray cells -> classify each cell crop -> build bill
```

## Created Artifacts

Annotation/layout:

```text
data/annotations/tray_cells.json
data/generated/annotation_previews/
```

Synthetic dataset:

```text
data/generated/tray_cell_synthetic/
```

Models:

```text
ml/models/detector/cell_best.pt
ml/models/cnn/cell_best.pt
```

Demo image:

```text
data/raw/real_trays_test/sample.jpg
```

## Verification Commands

End-to-end CLI:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe ml\inference\run_cell_pipeline.py data\raw\real_trays_test\sample.jpg --cell-detector-model ml\models\detector\cell_best.pt --cnn-model ml\models\cnn\cell_best.pt --json
```

Verified backend handler:

```text
cells 5
total 105000
items ['Trứng chiên', 'Trứng chiên', 'Thịt kho trứng', 'Đậu hũ sốt cà']
```

## Current Bootstrap Metrics

Synthetic validation after retraining from 500 generated tray images at `--confidence 0.05`:

```text
detection_recall: 1.0000
detection_precision: 1.0000
mean_iou: 0.9654
label_accuracy: 0.9800
```

These models are much stronger on synthetic data. They are still not final production-quality models for real camera images.

## Important Notes

- The current tray-cell layout was bootstrapped from `data/raw/empty_trays/01458c347bdb07f4f1c1269e26a11011.jpg`.
- The demo image is synthetic, generated from the same bootstrap layout.
- Add real top-down tray images with food and annotate them in `data/annotations/tray_cells.json` to improve real-world accuracy.
- Retrain `ml/models/detector/cell_best.pt` and `ml/models/cnn/cell_best.pt` after adding real annotated data.
