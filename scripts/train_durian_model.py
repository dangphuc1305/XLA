"""
Mã nguồn huấn luyện mô hình Deep Learning nhận diện bệnh lá sầu riêng:
- Kiến trúc: MobileNetV3-Small (Transfer Learning từ ImageNet)
- Bộ dữ liệu: Mendeley Durian Leaf Dataset v5 (2,595 ảnh, 6 lớp)
- Đánh giá: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Xuất mô hình: PyTorch (.pth) và ONNX (.onnx)
"""

import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Thiết lập đường dẫn
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset" / "mendeley_durian" / "Durian_Leaf_Diseases_Release_20261017_v2"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PTH = MODEL_DIR / "mobilenetv3_durian.pth"
MODEL_ONNX = MODEL_DIR / "mobilenetv3_durian.onnx"
METRICS_JSON = MODEL_DIR / "evaluation_metrics.json"

PRETRAINED_WEIGHTS_FILE = MODEL_DIR / "mobilenet_v3_small-047dcff4.pth"

CLASSES = [
    "Leaf_Algal",
    "Leaf_Blight",
    "Leaf_Colletotrichum",
    "Leaf_Healthy",
    "Leaf_Phomopsis",
    "Leaf_Rhizoctonia"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    try:
        import os
        num_cpus = os.cpu_count() or 4
        torch.set_num_threads(min(8, num_cpus))
        print(f"[CPU ACCEL] Đã cấu hình {torch.get_num_threads()} CPU threads cho PyTorch", flush=True)
    except Exception:
        pass


def get_dataloaders(batch_size=32):
    """Chuẩn bị DataLoaders cho Train, Val, Test với Data Augmentation."""
    # Data Augmentation chuẩn hóa cho ảnh thực vật
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(str(DATASET_DIR / "train"), transform=train_transforms)
    val_ds = datasets.ImageFolder(str(DATASET_DIR / "val"), transform=val_test_transforms)
    test_ds = datasets.ImageFolder(str(DATASET_DIR / "test"), transform=val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes=6):
    """Xây dựng MobileNetV3-Small với Transfer Learning từ file local."""
    model = mobilenet_v3_small(weights=None)
    if PRETRAINED_WEIGHTS_FILE.exists():
        print(f"[PRETRAINED] Tải trọng số khởi tạo từ: {PRETRAINED_WEIGHTS_FILE}", flush=True)
        pretrained_dict = torch.load(PRETRAINED_WEIGHTS_FILE, map_location=DEVICE)
        model.load_state_dict(pretrained_dict)
    else:
        print("[PRETRAINED] Cảnh báo: Không tìm thấy file trọng số local, dùng khởi tạo ngẫu nhiên.", flush=True)
    
    # Thay thế lớp classifier cuối cùng cho 6 lớp bệnh lá sầu riêng
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, epochs=6):
    """Huấn luyện mô hình với Early Stopping và lưu trọng số tốt nhất."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights = None

    print(f"\nBắt đầu huấn luyện mô hình MobileNetV3 trên thiết bị: {DEVICE}", flush=True)
    print(f"Tổng số Epochs: {epochs} | Batch size: {train_loader.batch_size}", flush=True)
    print("-" * 65, flush=True)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10}", flush=True)
    print("-" * 65, flush=True)

    for epoch in range(1, epochs + 1):
        # Phase Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        batch_idx = 0
        total_batches = len(train_loader)
        for images, labels in train_loader:
            batch_idx += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

            if batch_idx % 15 == 0 or batch_idx == total_batches:
                print(f"  -> Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Step Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        train_epoch_loss = train_loss / total_train
        train_epoch_acc = train_correct / total_train * 100.0

        # Phase Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_epoch_loss = val_loss / total_val
        val_epoch_acc = val_correct / total_val * 100.0

        print(f"{epoch:<8} | {train_epoch_loss:<12.4f} | {train_epoch_acc:<9.2f}% | {val_epoch_loss:<10.4f} | {val_epoch_acc:<9.2f}%", flush=True)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_weights = model.state_dict().copy()

    print("-" * 65, flush=True)
    print(f"[OK] Huấn luyện hoàn tất! Độ chính xác Validation cao nhất: {best_val_acc:.2f}%\n", flush=True)
    if best_weights:
        model.load_state_dict(best_weights)
        torch.save(best_weights, MODEL_PTH)
        print(f"[LƯU MÔ HÌNH] Trọng số tốt nhất đã lưu vào: {MODEL_PTH}", flush=True)

    return model


def evaluate_on_test(model, test_loader, class_names):
    """Đánh giá toàn diện trên tập Test độc lập và đo độ trễ suy luận."""
    model.eval()
    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            for single_img in images:
                t0 = time.perf_counter()
                out = model(single_img.unsqueeze(0))
                latencies.append((time.perf_counter() - t0) * 1000.0)
                _, pred = torch.max(out, 1)
                all_preds.append(pred.item())
            all_targets.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_latency = float(np.mean(latencies))

    # Báo cáo phân loại chi tiết
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    print("==================================================")
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (389 ẢNH):")
    print("==================================================")
    print(report_text)
    print(f"Thời gian suy luận trung bình mỗi ảnh (Inference Latency): {avg_latency:.2f} ms")
    print(f"Ma trận nhầm lẫn (Confusion Matrix):\n{np.array(cm)}")
    print("==================================================")

    # Lưu metrics vào JSON
    eval_data = {
        "overall_accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "average_latency_ms": round(avg_latency, 2),
        "classes": class_names,
        "classification_report": report,
        "confusion_matrix": cm,
        "test_samples_count": len(all_targets)
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Báo cáo đánh giá đã lưu vào: {METRICS_JSON}")


def export_to_onnx(model):
    """Xuất mô hình sang định dạng ONNX để triển khai siêu nhẹ."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_ONNX),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[ONNX EXPORT] Đã xuất mô hình ONNX thành công: {MODEL_ONNX} ({MODEL_ONNX.stat().st_size / (1024*1024):.1f} MB)")


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=32)
    model = build_model(num_classes=len(class_names))
    model = train_model(model, train_loader, val_loader, epochs=6)
    evaluate_on_test(model, test_loader, class_names)
    export_to_onnx(model)


if __name__ == "__main__":
    main()
