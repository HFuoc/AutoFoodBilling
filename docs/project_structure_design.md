# Food Recognition Pipeline Project Structure

## Understanding Summary

- Build a full food recognition pipeline for the Image Recogniting Challenge 2025.
- The pipeline must detect and crop food regions from a tray image, classify each crop with a CNN, then calculate the bill.
- The current dataset contains 11 food classes matching the PDF menu.
- The available food dataset is mostly single-food/cropped images, so it is suitable for CNN training.
- detector will be trained as a one-class detector named `food_item`, not as an 11-class food classifier.
- Empty tray images will be used as realistic backgrounds for synthetic detector training data.
- Runtime camera input is expected to be fixed top-down, reducing detection complexity.

## Assumptions

- The first implementation targets a local demo/research workflow, not production deployment.
- The CNN is responsible for recognizing the 11 dish classes.
- detector is responsible only for locating food regions to crop.
- The current `dataset/` folder will be reorganized into `data/raw/food_classes/`.
- Empty tray images will be stored in `data/raw/empty_trays/`.
- Real tray images, if collected later, will be stored in `data/raw/real_trays_test/` or a future annotation folder.
- CNN train/validation split must avoid duplicate leakage because the current dataset contains many copied files.

## Final Project Structure

```text
D:\FinalAI
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/
│  │  ├─ services/
│  │  │  ├─ detector.py
│  │  │  ├─ classifier.py
│  │  │  └─ billing.py
│  │  ├─ core/
│  │  └─ schemas/
│  ├─ tests/
│  └─ requirements.txt
│
├─ frontend/
│  ├─ src/
│  ├─ public/
│  └─ package.json
│
├─ ml/
│  ├─ training/
│  │  ├─ train_cnn.py
│  │  ├─ train_detector.py
│  │  └─ generate_detector_dataset.py
│  ├─ inference/
│  │  └─ run_pipeline.py
│  ├─ models/
│  │  ├─ cnn/
│  │  └─ detector/
│  └─ configs/
│
├─ data/
│  ├─ raw/
│  │  ├─ food_classes/
│  │  ├─ empty_trays/
│  │  └─ real_trays_test/
│  ├─ generated/
│  │  ├─ detector_synthetic/
│  │  └─ crops/
│  └─ processed/
│
├─ configs/
│  ├─ menu.json
│  ├─ classes.json
│  └─ paths.yaml
│
├─ scripts/
├─ notebooks/
├─ docs/
└─ README.md
```

## Data Flow

```text
Top-down tray image
  -> detector food_item detector
  -> cropped food images
  -> CNN food classifier
  -> menu price lookup
  -> console bill / backend API response / frontend display
```

## Decision Log

| Decision | Alternatives Considered | Reason |
| --- | --- | --- |
| Use full pipeline detector crop + CNN classify + billing | CNN-only classifier, detector-only classifier | Matches the PDF requirement and keeps detection/classification responsibilities clean. |
| Train detector with one class: `food_item` | Train detector with 11 food classes | One-class detection needs less tray data and lets CNN handle fine-grained dish recognition. |
| Use empty tray images as synthetic backgrounds | Require many real trays with all dishes | Real tray datasets are hard to collect; synthetic generation can bootstrap detector. |
| Keep `backend`, `frontend`, `ml`, and `data` separate | Put training scripts inside backend | Separating ML workflows from app/API code keeps the project easier to maintain. |
| Store prices in `configs/menu.json` | Hard-code prices in Python | A config file is easier to update and can be shared by inference/backend/frontend. |
| Split CNN data carefully by duplicate/hash groups | Random split by file | The current dataset has many copied files, so random file split would inflate validation accuracy. |

