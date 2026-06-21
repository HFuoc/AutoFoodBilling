# Frontend

Static Samsung-inspired demo UI for the tray-cell billing pipeline.

## Run

Start the backend API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend from `frontend/`:

```powershell
cd frontend
..\.venv\Scripts\python.exe -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173/index.html
```

Click **Dùng ảnh demo** or choose your own tray image.

## Camera Mode

Click **Mở camera** after plugging in a webcam. The browser will ask for camera permission, then the camera dropdown in the header will list available laptop and external cameras. Choose another camera from the dropdown to switch streams.

The UI captures one frame about every 1.8 seconds and sends it to:

```text
POST http://127.0.0.1:8000/predict-cells
```

The latest frame stays on screen with detected tray-cell boxes, predicted labels, and the bill total.

## Training Accuracy

The UI can show model output, but accuracy depends on corrected training data. Use:

```text
docs/manual_accuracy_training.md
```

Add clean tray images, annotate boxes and labels in `data/annotations/tray_cells.json`, then retrain both `cell_best.pt` files.
