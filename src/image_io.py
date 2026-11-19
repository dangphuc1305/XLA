"""Mã hóa/lưu ảnh OpenCV, hỗ trợ đường dẫn tiếng Việt trên Windows."""

from pathlib import Path

import cv2
import numpy as np


def encode_image(image: np.ndarray, extension: str = '.png') -> bytes:
    """Mã hóa ảnh BGR hoặc mask; PNG giữ nguyên giá trị nhị phân 0/255."""
    try:
        success, encoded = cv2.imencode(extension, image)
    except cv2.error as exc:
        raise OSError(f'Không mã hóa được ảnh dạng {extension}.') from exc
    if not success:
        raise OSError(f'Không mã hóa được ảnh dạng {extension}.')
    return encoded.tobytes()


def save_image(path, image: np.ndarray) -> None:
    """Ghi ảnh và báo lỗi thật nếu không ghi được, thay vì bỏ qua cờ imwrite.

    Thư mục cha phải tồn tại. Path.write_bytes hỗ trợ tên tệp Unicode;
    phần mã hóa vẫn do OpenCV đảm nhiệm, không đổi thứ tự kênh BGR.
    """
    path = Path(path)
    path.write_bytes(encode_image(image, path.suffix.lower()))
