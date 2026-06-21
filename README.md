# AutoFoodBilling

AutoFoodBilling là hệ thống nhận diện món ăn trên khay cơm và tự động tính tiền theo bảng giá cấu hình sẵn. Dự án kết hợp API FastAPI, giao diện demo trên trình duyệt và các script phục vụ huấn luyện/đánh giá mô hình nhận diện ảnh.

Luồng xử lý chính:

```text
Ảnh khay cơm -> xác định từng ô/khay món -> phân loại món bằng CNN -> tra giá trong menu -> xuất hóa đơn
```

## Mục Tiêu Dự Án

- Nhận diện các món ăn phổ biến trong khay cơm từ ảnh chụp top-down.
- Tính tổng tiền tự động dựa trên danh sách món đã nhận diện.
- Cung cấp giao diện demo để upload ảnh, dùng ảnh mẫu hoặc thử với camera.
- Hỗ trợ hai hướng xử lý:
  - `/predict`: pipeline cũ, xử lý ảnh và trả danh sách món.
  - `/predict-cells`: pipeline khuyến nghị, cắt từng ô/ngăn khay rồi phân loại món trong từng ô.
- Cung cấp script để kiểm tra dữ liệu, tạo dữ liệu synthetic, train CNN/detector và đánh giá pipeline.

## Công Nghệ Sử Dụng

- **Backend:** FastAPI, Uvicorn, Pydantic.
- **Xử lý ảnh:** Pillow, OpenCV.
- **Machine Learning:** PyTorch/Torchvision, Keras, ONNX Runtime.
- **Frontend:** HTML/CSS/JavaScript thuần, chạy bằng static HTTP server.
- **Cấu hình:** JSON/YAML cho class, menu, đường dẫn dữ liệu và model.

## Cấu Trúc Nhanh

```text
backend/       API FastAPI, schema response, service phân loại và tính tiền
frontend/      Giao diện demo upload ảnh/camera và hiển thị hóa đơn
ml/            Script huấn luyện, inference, đánh giá và thư mục model cục bộ
data/          Annotation và các thư mục dữ liệu cục bộ
configs/       Danh sách class, menu giá, đường dẫn mặc định
docs/          Tài liệu huấn luyện, checklist và ghi chú pipeline
scripts/       Script tiện ích cho kiểm tra dữ liệu và demo
h5/            Vị trí đặt model Keras HDF5 cục bộ khi chạy backend
```

Xem mô tả chi tiết từng thư mục và file quan trọng trong [CATALOG.md](CATALOG.md).

## Quy Ước Repo Sạch

Repo chỉ nên lưu mã nguồn, cấu hình, tài liệu, annotation cần thiết và ảnh demo nhỏ. Các thành phần sau không nên commit trực tiếp:

- Dataset ảnh thật trong `data/raw/`.
- Output sinh ra khi chạy API hoặc script trong `data/generated/`, `data/processed/`.
- Model/checkpoint binary như `.h5`, `.pt`, `.onnx`.
- Cache Python, môi trường ảo `.venv/`, thư mục build frontend.
- File đề bài/PDF tham khảo lớn không cần thiết cho source code.

Khi cần chạy inference, hãy đặt model đúng đường dẫn mà cấu hình đang dùng, ví dụ:

```text
h5/food_model.h5
h5/labels.json
```

`h5/labels.json` có thể được giữ trong repo nếu cần làm metadata nhẹ; file model `.h5` nên được lưu riêng hoặc chia sẻ qua release/artifact.

## Cài Đặt Môi Trường

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\FinalAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Nếu PowerShell chặn script, có thể chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Chuẩn Bị Dữ Liệu Và Model

Các thư mục dữ liệu mặc định:

```text
data/raw/food_classes/      Ảnh món ăn theo từng class để train CNN
data/raw/empty_trays/       Ảnh khay trống
data/raw/real_trays_test/   Ảnh khay có món để test pipeline
data/raw/tray_with_food/    Ảnh khay có món phục vụ annotation/train/test
```

Annotation chính:

```text
data/annotations/tray_cells.json
```

Model backend đang load mặc định:

```text
h5/food_model.h5
```

Nếu file model chưa tồn tại, backend vẫn khởi động được nhưng endpoint dự đoán sẽ lỗi khi cần load classifier. Hãy train/export model hoặc đặt model đã có vào đúng vị trí.

## Chạy Toàn Bộ Demo

Sau khi đã cài môi trường và chuẩn bị model, chạy:

```powershell
.\run_project.ps1
```

Hoặc double-click:

```text
run_project.bat
```

Script sẽ mở:

```text
Frontend: http://127.0.0.1:5173/index.html
Backend : http://127.0.0.1:8001
```

Muốn tắt project, đóng hai cửa sổ PowerShell backend/frontend.

## Chạy Riêng Backend Và Frontend

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
..\.venv\Scripts\python.exe -m http.server 5173
```

Mở trình duyệt:

```text
http://127.0.0.1:5173/index.html
```

## API Chính

Kiểm tra server:

```text
GET /health
```

Dự đoán theo pipeline cũ:

```text
POST /predict
```

Dự đoán theo từng ô khay:

```text
POST /predict-cells
```

Hai endpoint dự đoán nhận `multipart/form-data` với field:

```text
file = ảnh khay cơm
```

Response gồm đường dẫn ảnh upload, danh sách detection/cell prediction và `bill` chứa từng món, đơn giá, tiền tệ và tổng tiền.

## Cấu Hình Quan Trọng

- `configs/classes.json`: danh sách class món ăn chính.
- `configs/classes_with_empty.example.json`: ví dụ danh sách class có thêm nhãn `empty`.
- `configs/menu.json`: bảng giá theo từng label món ăn.
- `configs/paths.yaml`: đường dẫn mặc định cho dữ liệu, model và config.

Khi đổi tên class, cần cập nhật đồng bộ:

- Metadata/label map của model.
- `configs/classes.json`.
- `configs/menu.json`.
- Annotation trong `data/annotations/`.
- Các tài liệu hoặc script có hard-code label liên quan.

## Lệnh Hữu Ích

Kiểm tra số lượng ảnh và duplicate hash trong dataset:

```powershell
python scripts\check_dataset.py
```

Xem trước annotation ô khay:

```powershell
python ml\training\preview_tray_cell_annotations.py --annotations data\annotations\tray_cells.json
```

Kiểm tra annotation/crop:

```powershell
python ml\training\check_tray_cell_data.py --annotations data\annotations\tray_cells.json
```

Train CNN classifier:

```powershell
python ml\training\train_cnn.py --epochs 20 --batch-size 16
```

Train detector:

```powershell
python ml\training\train_detector.py --epochs 50 --batch-size 8
```

Chạy pipeline tray-cell bằng console:

```powershell
python ml\inference\run_cell_pipeline.py data\raw\real_trays_test\sample.jpg
```

Train nhanh cell CNN và mở demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_cell_cnn_and_demo.ps1
```

## Tài Liệu Liên Quan

- [CATALOG.md](CATALOG.md): catalog cấu trúc thư mục và file quan trọng.
- [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md): hướng dẫn chạy chi tiết theo từng bước.
- [docs/tray_cell_training.md](docs/tray_cell_training.md): workflow train tray-cell.
- [docs/manual_accuracy_training.md](docs/manual_accuracy_training.md): hướng dẫn cải thiện accuracy bằng annotation thủ công.
- [docs/manual_tray_cell_training_checklist.md](docs/manual_tray_cell_training_checklist.md): checklist train và kiểm tra.
- [docs/project_structure_design.md](docs/project_structure_design.md): ghi chú thiết kế cấu trúc dự án.

## Ghi Chú Vận Hành

- Frontend mặc định gọi API `/predict-cells`.
- `run_project.ps1` chạy backend ở port `8001`, còn ví dụ chạy riêng backend ở port `8000`.
- Nếu dùng `test_api.py`, hãy kiểm tra lại port trong file để khớp backend đang chạy.
- Nếu PowerShell hoặc console lỗi tiếng Việt, đặt biến môi trường:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## Kiểm Tra Trước Khi Push

Trước khi commit/push, nên chạy:

```powershell
git status --short
git ls-files data/raw
git ls-files "*.h5" "*.pt" "*.onnx"
```

Kết quả mong muốn:

- `data/raw` chỉ còn các file giữ chỗ như `.gitkeep` nếu cần.
- Không có checkpoint/model binary lớn trong danh sách tracked file.
- README và CATALOG đã trỏ đúng tên file, đúng port và đúng workflow hiện tại.
