"""Tái tạo minh chứng Nhựt: 10 ảnh, so sánh cũ/mới và thử đổi độ sáng.

Chạy từ thư mục dự án: python scripts/evaluate_spots.py
Không có mask đốm chuẩn: CSV chỉ mô tả đầu ra, không phải độ chính xác.
"""

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_image, preprocess_image
from src.leaf_segmentation import segment_leaf
from src.spot_detection import detect_spots
from src.image_io import save_image


def baseline_spots(image, leaf_mask):
    """Ngưỡng HSV [10,50,50]–[30,255,200] của mã trước khi sửa."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 50, 50), (30, 255, 200))
    mask = cv2.bitwise_and(mask, leaf_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Giữ nguyên cách xử lý cũ để so sánh, kể cả thiếu AND cuối hàm.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def annotate(image, mask):
    output = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)  # Đỏ trong BGR.
    return output


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, lighting, overview = [], [], []
    paths = sorted((ROOT / 'data').glob('*/*.png'))
    if not paths:
        raise FileNotFoundError('Không có ảnh PNG trong data/<nhóm>/.')
    for path in paths:
        image = preprocess_image(load_image(str(path)))
        cv2.setRNGSeed(0)  # Giữ khởi tạo GrabCut giống nhau giữa các lần chạy.
        leaf = segment_leaf(image)
        spots = detect_spots(image, leaf)
        old = baseline_spots(image, leaf)
        leaf_area = cv2.countNonZero(leaf)
        area = cv2.countNonZero(spots)
        outside = np.count_nonzero(spots[leaf == 0])
        # Kiểm tra các bất biến trên từng ảnh thật; không coi chỉ cần chạy
        # không lỗi là bằng chứng vùng đốm đã được phân đoạn chính xác.
        if spots.shape != leaf.shape or spots.dtype != np.uint8:
            raise AssertionError(f'Sai kích thước/dtype: {path}')
        if outside or not np.all((spots == 0) | (spots == 255)):
            raise AssertionError(f'Sai hợp đồng mask: {path}')
        row = {
            'image': path.relative_to(ROOT).as_posix(),
            'group': path.parent.name,
            'leaf_pixels': leaf_area,
            'baseline_spot_pixels': cv2.countNonZero(old),
            'new_spot_pixels': area,
            'baseline_percent': round(cv2.countNonZero(old) / leaf_area * 100, 4),
            'new_percent': round(area / leaf_area * 100, 4),
            'new_components': cv2.connectedComponents(spots)[0] - 1,
            'outside_leaf_pixels': int(outside),
            'mask_contract': 'PASS',
        }
        comparison.append(row)
        name = f'{path.parent.name}_{path.stem}'
        save_image(output_dir / f'{name}_leaf_mask.png', leaf)
        save_image(output_dir / f'{name}_spot_mask.png', spots)
        save_image(output_dir / f'{name}_analyzed.png', annotate(image, spots))

        # Thay đổi độ sáng SAU tiền xử lý, giữ nguyên mask lá để kiểm tra
        # riêng độ nhạy detector. Đây không phải ảnh chụp mới độc lập.
        for factor in (0.75, 1.0, 1.25):
            changed = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            changed_spots = detect_spots(changed, leaf)
            changed_outside = int(np.count_nonzero(changed_spots[leaf == 0]))
            if changed_outside:
                raise AssertionError(f'Đốm ngoài lá khi đổi sáng: {path}')
            lighting.append({
                'image': row['image'], 'brightness_factor': factor,
                'spot_pixels': cv2.countNonZero(changed_spots),
                'spot_percent': round(cv2.countNonZero(changed_spots) / leaf_area * 100, 4),
                'outside_leaf_pixels': changed_outside,
            })

        # Một hàng bốn cột: đầu vào / ngưỡng cũ / thuật toán mới / mask mới.
        tiles = []
        for label, tile in (
            (name, image), ('OLD HSV', annotate(image, old)),
            ('NEW HSV + LAB', annotate(image, spots)),
            ('NEW SPOT MASK', cv2.cvtColor(spots, cv2.COLOR_GRAY2BGR)),
        ):
            thumb = cv2.resize(tile, (260, 260), interpolation=cv2.INTER_AREA)
            header = np.full((28, 260, 3), 245, dtype=np.uint8)
            cv2.putText(header, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
            tiles.append(np.vstack((header, thumb)))
        overview.append(np.hstack(tiles))
        print(f"{row['image']}: {row['baseline_percent']:.2f}% -> {row['new_percent']:.2f}%, outside={outside}")

    write_csv(output_dir / 'comparison.csv', comparison)
    write_csv(output_dir / 'lighting_checks.csv', lighting)
    # Tách hai tấm để xem rõ hơn trên màn hình.
    for group in ('normal', 'spotted'):
        rows = [tile for tile, path in zip(overview, paths) if path.parent.name == group]
        if rows:
            save_image(output_dir / f'overview_{group}.png', np.vstack(rows))
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'nhut')
    evaluate(parser.parse_args().output)
