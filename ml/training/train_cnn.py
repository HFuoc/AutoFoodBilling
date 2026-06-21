from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.common import configure_utf8_stdout, iter_images, load_classes, project_path


def require_training_deps():
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image, ImageOps
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Install with: pip install -r backend/requirements.txt"
        ) from exc
    return torch, transforms, Image, ImageOps, DataLoader, Dataset, models


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str
    sha1: str


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_records(data_root: Path, class_names: list[str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label, class_name in enumerate(class_names):
        class_dir = data_root / class_name
        for path in iter_images(class_dir):
            records.append(
                ImageRecord(
                    path=path,
                    label=label,
                    class_name=class_name,
                    sha1=file_sha1(path),
                )
            )
    return records


def split_by_hash(
    records: list[ImageRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    train: list[ImageRecord] = []
    val: list[ImageRecord] = []

    class_names = sorted({record.class_name for record in records})
    for class_name in class_names:
        class_records = [record for record in records if record.class_name == class_name]
        groups: dict[str, list[ImageRecord]] = {}
        for record in class_records:
            groups.setdefault(record.sha1, []).append(record)
        group_values = list(groups.values())
        rng.shuffle(group_values)
        val_group_count = max(1, int(round(len(group_values) * val_ratio))) if len(group_values) > 1 else 0
        val_hashes = {group[0].sha1 for group in group_values[:val_group_count]}
        for group in group_values:
            if group[0].sha1 in val_hashes:
                val.extend(group)
            else:
                train.extend(group)

    return train, val


class FoodImageDataset:
    def __init__(self, records: list[ImageRecord], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        _, _, Image, ImageOps, _, _, _ = require_training_deps()
        record = self.records[index]
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), record.label


def build_model(model_name: str, num_classes: int, pretrained: bool):
    torch, _, _, _, _, _, models = require_training_deps()
    if model_name != "efficientnet_b0":
        raise ValueError(f"Unsupported model: {model_name}")
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.35),
        torch.nn.Linear(in_features, num_classes),
    )
    return model


def class_weights(records: list[ImageRecord], num_classes: int):
    torch, *_ = require_training_deps()
    counts = Counter(record.label for record in records)
    total = sum(counts.values())
    weights = []
    for label in range(num_classes):
        count = max(1, counts.get(label, 0))
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def sample_weights(records: list[ImageRecord]) -> list[float]:
    counts = Counter(record.label for record in records)
    return [1.0 / max(1, counts[record.label]) for record in records]


def accuracy(logits, labels) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def run_epoch(model, loader, criterion, optimizer, device: str, train: bool, phase: str):
    torch, *_ = require_training_deps()
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    total_batches = len(loader)
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += batch_size
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            running_loss = total_loss / max(1, total_count)
            running_acc = total_correct / max(1, total_count)
            print(
                f"{phase} batch {batch_index}/{total_batches} "
                f"loss={running_loss:.4f} acc={running_acc:.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
    }


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Train the CNN food classifier.")
    parser.add_argument("--data-root", default="data/raw/food_classes")
    parser.add_argument("--output", default="ml/models/cnn/best.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--classes", default="configs/classes.json")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch, transforms, _, _, DataLoader, _, _ = require_training_deps()
    from torch.utils.data import WeightedRandomSampler

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = project_path(args.data_root)
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading class list...", flush=True)
    class_names = load_classes(args.classes)
    print(f"Scanning training images in {data_root}...", flush=True)
    records = collect_records(data_root, class_names)
    if not records:
        raise SystemExit(f"No training images found in {data_root}")

    print(f"Found {len(records)} training images. Creating train/val split...", flush=True)
    train_records, val_records = split_by_hash(records, args.val_ratio, args.seed)
    if not train_records or not val_records:
        raise SystemExit("Unable to create train/val split. Add more unique images.")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.65, 1.0), ratio=(0.82, 1.22)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(8)], p=0.35),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.12)], p=0.2),
            transforms.ColorJitter(brightness=0.22, contrast=0.20, saturation=0.16, hue=0.02),
            transforms.RandomAutocontrast(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.18, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = FoodImageDataset(train_records, train_transform)
    val_dataset = FoodImageDataset(val_records, val_transform)
    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sampler = WeightedRandomSampler(
            weights=sample_weights(train_records),
            num_samples=len(train_records),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Building model: {args.model} pretrained={not args.no_pretrained}", flush=True)
    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_records, len(class_names)).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_accuracy = -1.0
    print(f"Classes: {len(class_names)}")
    print(f"Records: train={len(train_records)} val={len(val_records)}")
    print(f"Pretrained: {not args.no_pretrained}")
    print(f"Balanced sampler: {args.balanced_sampler}")

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}...", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            phase=f"epoch={epoch:03d} train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            phase=f"epoch={epoch:03d} val",
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            import json

            metadata = {
                "model_name": args.model,
                "class_names": class_names,
                "image_size": args.image_size,
                "val_accuracy": best_accuracy,
            }
            if output_path.suffix.lower() == ".onnx":
                dummy_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)
                model.eval()
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    dynamo=False,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                )
                meta_path = output_path.with_suffix(".json")
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                print(f"saved best model: {output_path} and {meta_path}", flush=True)
            else:
                torch.save(
                    {
                        **metadata,
                        "model_state": model.state_dict(),
                    },
                    output_path,
                )
                print(f"saved best model: {output_path}", flush=True)


if __name__ == "__main__":
    main()
