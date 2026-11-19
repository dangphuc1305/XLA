"""
Script tải nhanh bộ dữ liệu Mendeley bằng kỹ thuật Multi-thread HTTP Range:
182,056,031 bytes chia thành 8 luồng tải song song -> Tăng tốc gấp 8-10 lần!
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_URL = "https://data.mendeley.com/public-files/datasets/pxzvksbwnj/files/6de391cf-6f2b-45db-9634-243f5b278185/file_downloaded"
TOTAL_SIZE = 182056031
NUM_WORKERS = 8

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
ZIP_PATH = DATASET_DIR / "Durian_Leaf_Diseases.zip"
EXTRACT_DIR = DATASET_DIR / "mendeley_durian"


def download_chunk(part_idx, start_byte, end_byte):
    part_file = DATASET_DIR / f"part_{part_idx}.tmp"
    
    # Nếu file part đã tải đủ kích thước thì bỏ qua
    expected_part_size = end_byte - start_byte + 1
    if part_file.exists() and part_file.stat().st_size == expected_part_size:
        return part_idx, part_file

    req = urllib.request.Request(
        DATASET_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Range": f"bytes={start_byte}-{end_byte}",
            "Accept": "*/*"
        }
    )
    
    with urllib.request.urlopen(req, timeout=90) as resp, open(part_file, "wb") as f:
        while True:
            buf = resp.read(256 * 1024)
            if not buf:
                break
            f.write(buf)
            
    return part_idx, part_file


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Kiểm tra nếu đã giải nén xong
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[OK] Dữ liệu đã có sẵn tại: {EXTRACT_DIR}")
        for item in sorted(EXTRACT_DIR.iterdir()):
            if item.is_dir():
                print(f" - {item.name:25}: {len(list(item.glob('*.*')))} ảnh")
        return

    # 2. Tải 8 phân đoạn song song
    print(f"Bắt đầu tải song song {NUM_WORKERS} luồng từ Mendeley Data ({TOTAL_SIZE/(1024*1024):.1f} MB)...")
    t0 = time.time()
    
    chunk_size = (TOTAL_SIZE + NUM_WORKERS - 1) // NUM_WORKERS
    tasks = []
    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end = min(TOTAL_SIZE - 1, (i + 1) * chunk_size - 1)
        tasks.append((i, start, end))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_chunk, idx, s, e): idx for idx, s, e in tasks}
        completed = 0
        for fut in as_completed(futures):
            idx, pfile = fut.result()
            completed += 1
            print(f" -> Hoàn thành luồng {idx+1}/{NUM_WORKERS} ({pfile.stat().st_size/(1024*1024):.1f} MB) [{completed}/{NUM_WORKERS}]")

    print(f"\n[OK] Tải toàn bộ các luồng xong trong {time.time()-t0:.1f} giây!")
    
    # 3. Hợp nhất các phân đoạn thành file ZIP hoàn chỉnh
    print("Đang ghép các phân đoạn thành tệp ZIP...")
    with open(ZIP_PATH, "wb") as out_zip:
        for idx in range(NUM_WORKERS):
            part_file = DATASET_DIR / f"part_{idx}.tmp"
            with open(part_file, "rb") as pf:
                while True:
                    buf = pf.read(1024 * 1024)
                    if not buf:
                        break
                    out_zip.write(buf)
            part_file.unlink(missing_ok=True)
            
    print(f"[OK] Tệp ZIP hợp nhất thành công: {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")

    # 4. Giải nén dữ liệu
    print("Đang giải nén bộ dữ liệu...")
    t1 = time.time()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"[OK] Giải nén thành công trong {time.time()-t1:.1f} giây vào: {EXTRACT_DIR}")

    # 5. Thống kê kết quả
    print("\nThống kê danh mục các lớp đã giải nén:")
    for item in sorted(EXTRACT_DIR.iterdir()):
        if item.is_dir():
            print(f" - {item.name:25}: {len(list(item.glob('*.*')))} ảnh")


if __name__ == "__main__":
    main()
