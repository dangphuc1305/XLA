from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("dataset/mendeley_durian/Durian_Leaf_Diseases_Release_20261017_v2")

splits = ["train", "val", "test"]
classes = ["Leaf_Healthy", "Leaf_Algal", "Leaf_Colletotrichum", "Leaf_Phomopsis", "Leaf_Rhizoctonia", "Leaf_Blight"]

total_counts = {c: 0 for c in classes}
split_counts = {s: {c: 0 for c in classes} for s in splits}
corrupted = []
dimensions = set()

for s in splits:
    split_dir = DATASET_ROOT / s
    for c in classes:
        cls_dir = split_dir / c
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.*"):
            try:
                with Image.open(img_path) as img:
                    dimensions.add(img.size)
                    split_counts[s][c] += 1
                    total_counts[c] += 1
            except Exception as e:
                corrupted.append((str(img_path), str(e)))

print("==================================================")
print("THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU MENDELEY DURIAN LEAF:")
print("==================================================")
header = f"{'Nhãn bệnh (Class)':<22} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Tổng':<6}"
print(header)
print("-" * len(header))

for c in classes:
    tr = split_counts["train"][c]
    va = split_counts["val"][c]
    te = split_counts["test"][c]
    tot = total_counts[c]
    print(f"{c:<22} | {tr:<6} | {va:<6} | {te:<6} | {tot:<6}")

print("-" * len(header))
tot_tr = sum(split_counts["train"].values())
tot_va = sum(split_counts["val"].values())
tot_te = sum(split_counts["test"].values())
grand_total = sum(total_counts.values())
print(f"{'TỔNG CỘNG':<22} | {tot_tr:<6} | {tot_va:<6} | {tot_te:<6} | {grand_total:<6}")
print("==================================================")
print(f"Kích thước ảnh ghi nhận: {dimensions}")
print(f"Số lượng file lỗi/hỏng: {len(corrupted)}")
if corrupted:
    for f, err in corrupted[:5]:
        print(f" - {f}: {err}")
