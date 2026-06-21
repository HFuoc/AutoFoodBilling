from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _missing_dependency(name: str) -> SystemExit:
    return SystemExit(
        f"Missing dependency: {name}. Install dependencies with: "
        "pip install -r backend/requirements.txt"
    )


class FoodClassifier:
    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        try:
            import torchvision.transforms as transforms
            import torch
        except ImportError as exc:
            raise _missing_dependency("torch/torchvision") from exc

        self.torch = torch
        self.transforms = transforms
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"CNN model not found: {self.model_path}")

        self.session = None
        self.input_name = None
        self.model = None
        self.is_keras = False
        
        if self.model_path.suffix.lower() == ".onnx":
            self._load_onnx_model()
        else:
            is_hdf5 = False
            if self.model_path.suffix.lower() == ".h5":
                try:
                    with self.model_path.open("rb") as f:
                        if f.read(4) == b"\x89HDF":
                            is_hdf5 = True
                except Exception:
                    pass
            if is_hdf5:
                self._load_keras_model()
            else:
                self._load_torch_checkpoint()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_onnx_model(self) -> None:
        try:
            import json
            import onnxruntime as ort
        except ImportError as exc:
            raise _missing_dependency("onnxruntime") from exc

        meta_path = self.model_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"CNN metadata not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        self.class_names = meta["class_names"]
        self.image_size = int(meta.get("image_size", 224))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _load_torch_checkpoint(self) -> None:
        from torchvision import models

        checkpoint = self.torch.load(self.model_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise FileNotFoundError(f"Unsupported CNN checkpoint format: {self.model_path}")

        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint.get("image_size", 224))
        model_name = str(checkpoint.get("model_name", "efficientnet_b0"))
        if model_name != "efficientnet_b0":
            raise FileNotFoundError(f"Unsupported CNN model architecture: {model_name}")

        state = checkpoint["model_state"]
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        if "classifier.1.weight" in state:
            model.classifier[1] = self.torch.nn.Linear(in_features, len(self.class_names))
        else:
            model.classifier[1] = self.torch.nn.Sequential(
                self.torch.nn.Dropout(p=0.35),
                self.torch.nn.Linear(in_features, len(self.class_names)),
            )
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def _load_keras_model(self) -> None:
        try:
            import os
            os.environ["KERAS_BACKEND"] = "torch"
            import keras
            import json
        except ImportError as exc:
            raise _missing_dependency("keras") from exc
        
        meta_path = self.model_path.with_name("labels.json")
        if not meta_path.exists():
            meta_path = self.model_path.with_suffix(".json")
            
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                if isinstance(meta, dict) and "0" in meta:
                    self.class_names = [meta[str(i)] for i in range(len(meta))]
                elif "class_names" in meta:
                    self.class_names = meta["class_names"]
                else:
                    self.class_names = list(meta.values())
        else:
            self.class_names = [str(i) for i in range(10)]
            
        self.model = keras.saving.load_model(str(self.model_path))
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # shape is usually (batch, height, width, channels)
        if len(input_shape) >= 3 and input_shape[1] is not None:
            self.image_size = input_shape[1]
        else:
            self.image_size = 128
        self.is_keras = True

    def predict_image(self, image: Any, top_k: int = 3) -> dict[str, Any]:
        image = image.convert("RGB")
        
        if self.is_keras:
            import numpy as np
            image_resized = image.resize((self.image_size, self.image_size))
            input_data = np.array(image_resized, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            logits = self.model.predict(input_data, verbose=0)[0]
            probs = self.torch.tensor(logits)
        else:
            tensor = self.transform(image).unsqueeze(0)
            if self.session is not None:
                import numpy as np
    
                input_data = tensor.numpy().astype(np.float32)
                logits = self.session.run(None, {self.input_name: input_data})[0]
                logits_tensor = self.torch.from_numpy(logits)
            else:
                assert self.model is not None
                with self.torch.no_grad():
                    logits_tensor = self.model(tensor.to(self.device)).cpu()
            probs = self.torch.softmax(logits_tensor, dim=1)[0]

        values, indices = self.torch.topk(probs, k=min(top_k, len(self.class_names), len(probs)))
        predictions = [
            {
                "label": self.class_names[int(index)],
                "confidence": float(value),
            }
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        return {
            "label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
        }

    def predict_path(self, image_path: str | Path, top_k: int = 3) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise _missing_dependency("pillow") from exc

        with Image.open(image_path) as image:
            return self.predict_image(image, top_k=top_k)
