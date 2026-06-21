# CATALOG Cấu Trúc Dự Án

Tài liệu này mô tả các thư mục và file quan trọng của AutoFoodBilling. Mục tiêu là giúp người mới vào dự án hiểu nhanh: mỗi thư mục dùng để làm gì, file nào cần quan tâm, file nào chỉ là dữ liệu cục bộ không nên commit.

## Nguyên Tắc Phân Loại File

- **Nên commit:** mã nguồn, cấu hình, tài liệu, annotation mẫu/chính, script chạy, ảnh demo nhỏ phục vụ UI.
- **Không nên commit:** dataset ảnh thật, output sinh ra khi chạy script/API, model/checkpoint binary, cache, môi trường ảo, file PDF tham khảo nặng.
- **Có thể giữ cục bộ:** `data/raw/`, `data/generated/`, `data/processed/`, `h5/*.h5`, `ml/models/**/*.h5`, `ml/models/**/*.pt`.

## Tổng Quan Cây Thư Mục

```text
AutoFoodBilling/
|-- backend/       API FastAPI và service xử lý nghiệp vụ
|-- frontend/      Giao diện demo chạy trên trình duyệt
|-- ml/            Script huấn luyện, inference, đánh giá model
|-- data/          Annotation và dữ liệu cục bộ
|-- configs/       Class, menu giá, đường dẫn mặc định
|-- docs/          Tài liệu kỹ thuật và checklist huấn luyện
|-- scripts/       Script tiện ích
|-- h5/            Vị trí đặt model Keras cục bộ
|-- README.md      Tài liệu tổng quan và hướng dẫn chạy
|-- CATALOG.md     Catalog cấu trúc dự án
`-- run_project.*  Script khởi động demo
```

## backend/

Thư mục backend chứa API FastAPI và các service dùng khi nhận ảnh, phân loại món, tính tiền và trả response cho frontend.

```text
backend/
|-- app/
|   |-- main.py
|   |-- schemas/
|   |   |-- __init__.py
|   |   `-- prediction.py
|   `-- services/
|       |-- __init__.py
|       |-- billing.py
|       |-- classifier.py
|       `-- tray_layout.py
|-- tests/
`-- requirements.txt
```

### File quan trọng

- `backend/app/main.py`: điểm vào của FastAPI. File này khai báo app, cấu hình CORS, nhận upload ảnh, tạo crop, gọi classifier, áp dụng các rule fallback/rescue và trả kết quả cho `/predict`, `/predict-cells`.
- `backend/app/schemas/prediction.py`: định nghĩa response model bằng Pydantic cho prediction thường và prediction theo từng ô khay.
- `backend/app/services/billing.py`: đọc `configs/menu.json`, tạo từng dòng hóa đơn và tính tổng tiền.
- `backend/app/services/classifier.py`: wrapper load model `.h5`, `.pt`, `.onnx`; chuẩn hóa ảnh đầu vào và trả nhãn/top-k confidence.
- `backend/app/services/tray_layout.py`: helper xác định layout khay và crop các ô theo template.
- `backend/requirements.txt`: danh sách thư viện Python cần cài để chạy backend, xử lý ảnh và inference.
- `backend/tests/`: vị trí dành cho test backend. Hiện mới có file giữ chỗ.

## frontend/

Frontend là static web app dùng để demo pipeline nhận diện và tính tiền. Người dùng có thể chọn ảnh, dùng ảnh demo hoặc mở camera.

```text
frontend/
|-- index.html
|-- README.md
|-- public/
|   |-- demo-tray.png
|   `-- demo-tray-test.jpg
`-- src/
    |-- app.js
    `-- styles.css
```

### File quan trọng

- `frontend/index.html`: cấu trúc HTML chính của giao diện.
- `frontend/src/app.js`: xử lý tương tác upload/camera, gọi API `/predict-cells`, render bounding box, label, confidence và bill.
- `frontend/src/styles.css`: style của giao diện demo.
- `frontend/public/demo-tray.png`: ảnh demo chính, được phép commit vì phục vụ UI và smoke test.
- `frontend/public/demo-tray-test.jpg`: ảnh test nhỏ phục vụ demo.
- `frontend/README.md`: hướng dẫn chạy riêng frontend và mô tả camera mode.

## ml/

Thư mục `ml/` chứa code phục vụ huấn luyện, inference dòng lệnh và đánh giá pipeline.

```text
ml/
|-- __init__.py
|-- common.py
|-- configs/
|-- inference/
|-- models/
`-- training/
```

### File và thư mục quan trọng

- `ml/common.py`: helper dùng chung, đặc biệt cho việc resolve đường dẫn project/config.
- `ml/inference/run_pipeline.py`: chạy pipeline cũ bằng console.
- `ml/inference/run_cell_pipeline.py`: chạy pipeline nhận diện từng ô khay bằng console.
- `ml/inference/evaluate_cell_pipeline.py`: đánh giá kết quả tray-cell dựa trên annotation.
- `ml/training/train_cnn.py`: train CNN classifier từ dataset ảnh món/crop.
- `ml/training/train_detector.py`: train detector theo cấu hình dataset YOLO.
- `ml/training/train_tray_cell_detector.py`: train detector chuyên cho ô khay.
- `ml/training/generate_detector_dataset.py`: tạo dataset detector synthetic cho workflow cũ.
- `ml/training/generate_synthetic_tray_cell_dataset.py`: tạo dữ liệu synthetic cho workflow tray-cell.
- `ml/training/bootstrap_tray_cell_models.py`: chạy chuỗi bước bootstrap model tray-cell.
- `ml/training/prepare_tray_cell_detector_dataset.py`: chuyển annotation khay thật thành dataset detector.
- `ml/training/extract_tray_cell_crops.py`: cắt crop từng ô khay từ annotation để train CNN.
- `ml/training/check_tray_cell_data.py`: kiểm tra annotation, crop và phân bố label.
- `ml/training/preview_tray_cell_annotations.py`: tạo preview kiểm tra box annotation.
- `ml/training/merge_detector_datasets.py`: gộp dataset detector.
- `ml/training/merge_cnn_datasets.py`: gộp dataset CNN.
- `ml/training/prepare_supplemented_cnn_dataset.py`: chuẩn bị dataset CNN đã bổ sung.
- `ml/training/seed_tray_cell_annotations.py`: tạo annotation seed.
- `ml/training/seed_fixed_tray_annotations.py`: tạo annotation template cố định.
- `ml/models/`: nơi đặt model cục bộ sau khi train. Không nên commit checkpoint/model binary.

## data/

Thư mục dữ liệu gồm annotation và các vùng dữ liệu cục bộ. Repo chỉ nên giữ annotation cần thiết và file `.gitkeep`; dataset ảnh thật nên để ngoài Git.

```text
data/
|-- annotations/
|-- raw/
|-- generated/
`-- processed/
```

### data/annotations/

- `tray_cells.json`: annotation chính cho ảnh khay thật, gồm đường dẫn ảnh, box từng ô và label.
- `tray_cells.example.json`: ví dụ format annotation.
- `tray_cells_auto.json`: annotation được sinh tự động, dùng để tham khảo hoặc rà soát.
- `tray_cells_fixed_template.json`: template/toạ độ cố định cho một số layout khay.
- `tray_layouts_empty.json`: thông tin layout khay trống.

### data/raw/

Chứa dữ liệu ảnh gốc cục bộ. Không nên upload toàn bộ lên GitHub.

- `food_classes/`: ảnh món ăn theo từng class để train CNN.
- `empty_trays/`: ảnh khay trống.
- `real_trays_test/`: ảnh khay thật để test pipeline.
- `tray_with_food/`: ảnh khay có món phục vụ annotation/train/test.

Trong repo sạch, các thư mục này chỉ cần file `.gitkeep` nếu muốn giữ cấu trúc thư mục.

### data/generated/

Output sinh ra khi chạy API hoặc script, ví dụ crop, preview, dataset synthetic. Luôn ignore.

### data/processed/

Dataset đã xử lý để train detector/CNN. Luôn ignore vì có thể tái tạo từ dữ liệu gốc và script.

## configs/

Chứa cấu hình nhỏ nhưng rất quan trọng. Các file này nên được commit.

```text
configs/
|-- classes.json
|-- classes_with_empty.example.json
|-- menu.json
`-- paths.yaml
```

- `classes.json`: danh sách class món ăn chính.
- `classes_with_empty.example.json`: ví dụ danh sách class có nhãn `empty`.
- `menu.json`: bảng giá theo label món ăn, dùng trực tiếp khi tính hóa đơn.
- `paths.yaml`: đường dẫn mặc định cho dữ liệu, model và config.

## docs/

Chứa tài liệu kỹ thuật hỗ trợ train và vận hành.

- `docs/tray_cell_training.md`: hướng dẫn workflow nhận diện từng ô khay.
- `docs/manual_accuracy_training.md`: cách bổ sung/correct data thủ công để cải thiện accuracy.
- `docs/manual_tray_cell_training_checklist.md`: checklist huấn luyện và kiểm tra.
- `docs/pipeline_bootstrap_status.md`: ghi chú trạng thái bootstrap pipeline.
- `docs/project_structure_design.md`: ghi chú thiết kế cấu trúc dự án.

## scripts/

Chứa script tiện ích chạy từ root project.

- `scripts/check_dataset.py`: kiểm tra số lượng ảnh theo class và duplicate hash.
- `scripts/debug_canh_rau.py`: hỗ trợ debug class `Canh rau`.
- `scripts/train_cell_cnn_and_demo.ps1`: train nhanh cell CNN, smoke test và mở demo.

## h5/

Vị trí đặt model Keras HDF5 khi chạy backend.

- `h5/food_model.h5`: model classifier cục bộ, không nên commit.
- `h5/labels.json`: label map nhẹ, có thể commit nếu cần đồng bộ class index.

## File Ở Root

- `README.md`: tài liệu tổng quan, setup, cách chạy và quy ước repo.
- `CATALOG.md`: tài liệu hiện tại, mô tả cấu trúc thư mục/file.
- `HUONG_DAN_CHAY.md`: hướng dẫn chạy chi tiết theo từng bước.
- `run_project.ps1`: script PowerShell khởi động backend và frontend.
- `run_project.bat`: wrapper để double-click chạy `run_project.ps1`.
- `run_training.bat`: wrapper phục vụ train nhanh theo workflow đã cấu hình.
- `test_api.py`: script test nhanh endpoint bằng ảnh demo; cần kiểm tra port trước khi dùng.
- `.gitignore`: quy định file/thư mục không được đưa vào repo.

## Checklist Trước Khi Commit

Chạy các lệnh sau để kiểm tra repo không còn file dư thừa:

```powershell
git status --short
git ls-files data/raw
git ls-files "*.h5" "*.pt" "*.onnx"
git ls-files | Select-String "__pycache__|data/generated|data/processed|\\.venv"
```

Kỳ vọng:

- `data/raw` chỉ còn `.gitkeep` hoặc không có file ảnh.
- Không còn model/checkpoint binary trong Git.
- Không còn cache/runtime output.
- `README.md` trỏ tới `CATALOG.md` đúng chữ hoa.
