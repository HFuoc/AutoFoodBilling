# Huong Dan Chay Project

File nay huong dan chay pipeline:

```text
Anh khay com -> detector crop mon an -> CNN nhan dang mon -> tinh tien
```

## 1. Tao Moi Truong

Mo PowerShell tai thu muc project:

```powershell
cd D:\FinalAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

## Chay Tat Ca Bang 1 Lenh

Sau khi da setup `.venv`, mo PowerShell tai thu muc project:

```powershell
cd D:\FinalAI
.\run_project.ps1
```

Hoac double-click file:

```text
run_project.bat
```

Lenh nay se tu khoi dong backend port `8001`, frontend port `5173`, va mo trinh duyet vao:

```text
http://127.0.0.1:5173/index.html
```

Muon tat project thi dong 2 cua so PowerShell backend/frontend.

## Train Cell CNN Roi Mo Frontend Demo

Frontend hien goi endpoint:

```text
POST http://127.0.0.1:8001/predict-cells
```

Endpoint nay can model:

```text
ml/models/cnn/cell_best.cnn
```

Chay pipeline train va mo demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_cell_cnn_and_demo.ps1
```

Lenh tren se:

- Extract crop tu `data/annotations/tray_cells_fixed_template.json`
- Tron crop that voi anh mon an trong `data/raw/food_classes`
- Train CNN cell classifier va luu checkpoint `.cnn` dung duong dan backend dang dung
- Smoke test bang `frontend/public/demo-tray.png`
- Mo backend port `8001` va frontend port `5173`

Muon xem lenh ma chua train:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_cell_cnn_and_demo.ps1 -DryRun -NoBrowser
```

Muon train nhanh de test luong chay truoc:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_cell_cnn_and_demo.ps1 -Epochs 1 -NoBrowser
```

## 2. Kiem Tra Dataset Mon An

Dataset 11 mon hien nam o:

```text
data/raw/food_classes/
```

Chay lenh kiem tra so luong anh:

```powershell
python scripts\check_dataset.py
```

## 3. Them Anh Khay Trong

Dat anh khay trong chup tu tren xuong vao thu muc:

```text
data/raw/empty_trays/
```

Nen chup nhieu anh khac nhau ve anh sang, vi tri khay, goc lech nhe. Camera khi demo nen cung kieu top-down nhu anh khay trong.

## 4. Tao Dataset detector Synthetic

Sau khi da co anh khay trong:

```powershell
python ml\training\generate_detector_dataset.py --samples 800 --overwrite
```

Ket qua se duoc tao o:

```text
data/generated/detector_synthetic/
```

File cau hinh train detector:

```text
data/generated/detector_synthetic/data.yaml
```

## 5. Train CNN Phan Loai Mon An

```powershell
python ml\training\train_cnn.py --epochs 20 --batch-size 16
```

Model tot nhat se luu tai:

```text
ml/models/cnn/best.pt
```

## 6. Train detector Crop Mon An

```powershell
python ml\training\train_detector.py --epochs 50 --batch-size 8
```

Model detector tot nhat se luu tai:

```text
ml/models/detector/best.pt
```

## 7. Chay Full Pipeline Bang Console

Dat anh khay com that vao:

```text
data/raw/real_trays_test/
```

Vi du co file:

```text
data/raw/real_trays_test/sample.jpg
```

Chay:

```powershell
python ml\inference\run_pipeline.py data\raw\real_trays_test\sample.jpg
```

Ket qua se in ra console gom:

- Ten tung mon duoc nhan dang
- Gia tien tung mon
- Tong hoa don

Anh crop se duoc luu o:

```text
data/generated/crops/
```

## 8. Chay Backend API

Khoi dong API:

```powershell
uvicorn backend.app.main:app --reload
```

Kiem tra API:

```text
GET http://127.0.0.1:8000/health
```

Du doan anh:

```text
POST http://127.0.0.1:8000/predict
```

Gui form-data voi field:

```text
file = anh khay com
```

Neu dung workflow nhan dien tung o trong khay:

```text
POST http://127.0.0.1:8000/predict-cells
```

## 9. Thu Tu Chay Khuyen Nghi

```text
1. Them anh khay trong vao data/raw/empty_trays/
2. python scripts/check_dataset.py
3. python ml/training/generate_detector_dataset.py --samples 800 --overwrite
4. python ml/training/train_cnn.py --epochs 20 --batch-size 16
5. python ml/training/train_detector.py --epochs 50 --batch-size 8
6. python ml/inference/run_pipeline.py data/raw/real_trays_test/sample.jpg
```

## 9.1. Huong Moi: Nhan Dien Tung O Trong Khay

Neu muc tieu la nhan dien tung o/ngan cua khay truoc, kiem tra annotation truoc:

```powershell
python ml\training\preview_tray_cell_annotations.py --annotations data\annotations\tray_cells.json
python ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json --skip-crops
```

Neu chua co nhieu anh khay that, co the bootstrap bang synthetic data:

```powershell
python ml\training\bootstrap_tray_cell_models.py --layouts data\annotations\tray_cells.json --samples 300 --overwrite
```

Hoac chay tung buoc:

```powershell
python ml\training\generate_synthetic_tray_cell_dataset.py --layouts data\annotations\tray_cells.json --samples 300 --overwrite
python ml\training\train_detector.py --data data\generated\tray_cell_synthetic\detector\data.yaml --name tray_cell_synthetic --output ml\models\detector\cell_best.pt
python ml\training\train_cnn.py --data-root data\generated\tray_cell_synthetic\cnn --classes data\generated\tray_cell_synthetic\cnn\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16
python ml\inference\evaluate_cell_pipeline.py --annotations data\generated\tray_cell_synthetic\metadata\synthetic_cells_val.json --cell-detector-model ml\models\detector\cell_best.pt --cnn-model ml\models\cnn\cell_best.pt
```

Khi da co annotation khay that, train bang real data:

```powershell
python ml\training\prepare_tray_cell_detector_dataset.py --overwrite
python ml\training\train_detector.py --data data\processed\tray_cells_detector\data.yaml --name tray_cell --output ml\models\detector\cell_best.pt
python ml\training\extract_tray_cell_crops.py --overwrite
python ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json
python ml\training\train_cnn.py --data-root data\processed\tray_cell_food_classes --classes data\processed\tray_cell_food_classes\classes.json --output ml\models\cnn\cell_best.pt --epochs 20 --batch-size 16
python ml\inference\run_cell_pipeline.py data\raw\real_trays_test\sample.jpg
python ml\inference\evaluate_cell_pipeline.py --annotations data\annotations\tray_cells.json --cell-detector-model ml\models\detector\cell_best.pt --cnn-model ml\models\cnn\cell_best.pt
```

Can tao file annotation:

```text
data/annotations/tray_cells.json
```

Xem chi tiet o:

```text
docs/tray_cell_training.md
```

## 10. Loi Thuong Gap

### Thieu anh khay trong

Neu gap loi:

```text
No empty tray images found
```

Hay them anh vao:

```text
data/raw/empty_trays/
```

### Chua co model

Neu gap loi khong thay:

```text
ml/models/cnn/best.pt
ml/models/detector/best.pt
```

Hay train CNN va detector truoc khi chay inference.

### Loi thieu thu vien

Chay lai:

```powershell
pip install -r backend\requirements.txt
```

### Console loi tieng Viet

Neu PowerShell bi loi font tieng Viet, chay:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

roi chay lai lenh Python.
