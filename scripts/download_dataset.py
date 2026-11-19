"""
Script tải và giải nén bộ dữ liệu Mendeley:
A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis
DOI: 10.17632/pxzvksbwnj.5
File: Durian_Leaf_Diseases_Release_20261017_v2.zip (173.6 MB)
"""

import os
import sys
import time
import zipfile
import hashlib
import urllib.request
from pathlib import Path

DATASET_URL = "https://data.mendeley.com/public-files/datasets/pxzvksbwnj/files/6de391cf-6f2b-45db-9634-243f5b278185/file_downloaded"
EXPECTED_SHA256 = "98e2c23fdc67a0c9e624d7431fbcf5c56377074041fdb3b0872148c1b38ebd9c"
EXPECTED_SIZE = 182056031

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
ZIP_PATH = DATASET_DIR / "Durian_Leaf_Diseases.zip"
EXTRACT_DIR = DATASET_DIR / "mendeley_durian"


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        msg = f"\r[Tải dữ liệu] {percent:5.1f}% ({mb_down:6.1f}MB / {mb_total:6.1f}MB)"
    else:
        mb_down = downloaded / (1024 * 1024)
        msg = f"\r[Tải dữ liệu] {mb_down:6.1f}MB"
    sys.stdout.write(msg)
    sys.stdout.flush()


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra nếu đã giải nén xong
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[OK] Dữ liệu đã được giải nén sẵn tại: {EXTRACT_DIR}")
        return

    # Tải file ZIP nếu chưa có hoặc kích thước sai
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size != EXPECTED_SIZE:
        print(f"Bắt đầu tải bộ dữ liệu từ Mendeley Data (173.6 MB)...")
        print(f"URL: {DATASET_URL}")
        t0 = time.time()
        
        req = urllib.request.Request(
            DATASET_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*"
            }
        )
        
        with urllib.request.urlopen(req, timeout=60) as resp, open(ZIP_PATH, 'wb') as out_file:
            total_size = int(resp.headers.get('content-length', EXPECTED_SIZE))
            downloaded = 0
            block_size = 64 * 1024
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                percent = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\r[Tải dữ liệu] {percent:5.1f}% ({downloaded/(1024*1024):5.1f}MB / {total_size/(1024*1024):5.1f}MB)")
                sys.stdout.flush()
        
        print(f"\n[OK] Tải hoàn tất trong {time.time()-t0:.1f} giây!")
    else:
        print(f"[OK] Tệp zip đã tồn tại ({ZIP_PATH.stat().st_size} bytes)")

    # Kiểm tra và giải nén
    print("Đang giải nén bộ dữ liệu...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"[OK] Giải nén thành công vào {EXTRACT_DIR} trong {time.time()-t0:.1f} giây!")
    
    # Thống kê nhanh các lớp
    print("\nThống kê các lớp dữ liệu đã giải nén:")
    for item in sorted(EXTRACT_DIR.iterdir()):
        if item.is_dir():
            count = len(list(item.glob("*.*")))
            print(f" - {item.name:25}: {count} tệp ảnh")


if __name__ == "__main__":
    main()
