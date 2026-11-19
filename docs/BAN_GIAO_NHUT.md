# Bàn giao phần Trần Minh Nhựt

Ngày hoàn thiện và rà soát: 11/11/2026. Phạm vi: phát hiện đốm bất thường, xuất mask và ảnh khoanh đỏ, so sánh kết quả, nội dung Chương 3 theo bảng phân công. Đã đồng bộ `main.py` và `app.py`; lần rà soát bổ sung chạy ảnh mẫu, tải kết quả và kiểm tra lỗi lưu ảnh. Phép tính mức độ của Phúc tiếp tục nhận hai mask.

## 1. Sản phẩm

| Sản phẩm | Đường dẫn |
| --- | --- |
| Hàm Python có chú thích chi tiết từng bước | `src/spot_detection.py` |
| Kiểm thử tự động trên ảnh tổng hợp | `tests/test_spot_detection.py` |
| Chương 3: phương pháp và thuật toán | [CHUONG_3_NHUT.md](CHUONG_3_NHUT.md) |
| Công cụ tái tạo kết quả | `scripts/evaluate_spots.py` |
| Ghi ảnh có kiểm tra lỗi, hỗ trợ đường dẫn tiếng Việt | `src/image_io.py` |
| Kiểm thử luồng giao diện và lưu ảnh | `tests/test_app.py`, `tests/test_image_io.py` |
| Bảng so sánh 10 ảnh cũ/mới | `results/nhut/comparison.csv` |
| Kiểm tra độ sáng 0,75 / 1 / 1,25 trên 10 ảnh | `results/nhut/lighting_checks.csv` |
| Mỗi ảnh có mask lá, mask đốm và ảnh khoanh đỏ | `results/nhut/<normal hoặc spotted>_<tên>_*.png` |
| Ảnh tổng hợp so sánh | `results/nhut/overview_normal.png`, `overview_spotted.png` |

Các tệp trong `results/nhut/` là kết quả mới. Không dùng tỷ lệ trong bảng kiểm thử Hào cũ để mô tả thuật toán sau khi thay đổi.

## 2. Hợp đồng dữ liệu bắt buộc

```python
spot_mask = detect_spots(processed_bgr, leaf_mask)
```

| Mảng | Kích thước, kiểu | Quy ước |
| --- | --- | --- |
| `processed_bgr` | `(H, W, 3)`, `np.uint8` | Ảnh màu **BGR của OpenCV**, đã tiền xử lý |
| `leaf_mask` | `(H, W)`, `np.uint8` | Trắng 255 = lá; đen 0 = nền |
| `spot_mask` | `(H, W)`, `np.uint8` | Trắng 255 = đốm; đen 0 = vùng khác |

Ba mảng có cùng chiều cao/rộng. Không truyền ảnh RGB, ảnh mask 3 kênh hay mask 0/1. Có thể truyền ảnh đã tô đen nền nhưng không bắt buộc. Hàm không đổi kích thước và không thay đổi mảng đầu vào. Mask lá toàn đen trả mask đốm toàn đen. Đầu vào sai quy cách sinh `TypeError` hoặc `ValueError` có giải thích.

Hàm trả **một mảng**, không còn trả `(spot_mask, contours)`. Ràng buộc tại cuối hàm:

```python
return cv2.bitwise_and(spot_mask, leaf_mask)
```

Điều kiện cần giữ ở mọi bước: `np.all(spot_mask[leaf_mask == 0] == 0)`.

## 3. Tích hợp cho Hào và Phúc

```python
import cv2
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity

spot_mask = detect_spots(processed_bgr, leaf_mask)

# Phúc nhận đúng hai mask, tính số_pixel_đốm / số_pixel_lá × 100.
metrics = calculate_severity(leaf_mask, spot_mask)

# Đường viền lấy sau khi lọc để ảnh hiển thị khớp diện tích tính toán.
contours, _ = cv2.findContours(
    spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
annotated_bgr = processed_bgr.copy()
cv2.drawContours(annotated_bgr, contours, -1, (0, 0, 255), 2)
```

Ảnh khoanh đỏ sử dụng ảnh BGR đã tiền xử lý vì ảnh này và mask nằm cùng hệ tọa độ. Nếu muốn vẽ lên ảnh gốc trước resize, phải đổi mask về kích thước gốc bằng `cv2.INTER_NEAREST` rồi mới tìm contour. Khi hiển thị ảnh màu bằng PIL/Streamlit, đổi BGR sang RGB; giữ nguyên mask một kênh.

## 4. Cách chạy trên máy hiện tại

Đã tạo môi trường `.venv` và cài các thư viện trong `requirements.txt`. Từ thư mục gốc dự án:

```powershell
# Kiểm tra thuật toán và hợp đồng dữ liệu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Tạo lại mask, ảnh khoanh đỏ và CSV cho 10 ảnh có sẵn
.\.venv\Scripts\python.exe scripts\evaluate_spots.py

# Phân tích một ảnh qua quy trình CLI
.\.venv\Scripts\python.exe main.py data\spotted\hu1.png --output results\nhut\cli_smoke

# Giao diện đã nối với hàm mới
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Trong giao diện, chọn **Ảnh mẫu có sẵn**, chọn ảnh rồi bấm **Phân tích ảnh**. Có thể chuyển sang **Tải ảnh của bạn** để dùng tệp riêng. Giao diện hiển thị bốn ảnh và số liệu, có mục tải mask lá, mask đốm, ảnh khoanh đỏ và CSV. Khi đổi ảnh, số liệu cũ được xóa. Ảnh tổng hợp kết quả chạy nằm tại `results/nhut/demo_results.png`.

Trên máy khác: tạo môi trường bằng `python -m venv .venv`, sau đó chạy `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## 5. Kết quả và giới hạn cần nói khi thuyết trình

- **Kiểm tra hoàn tất:** 21/21 kiểm thử tự động đạt. Streamlit AppTest đã chạy phân tích ảnh mẫu, đối chiếu số liệu, đổi ảnh không giữ kết quả cũ, xử lý tỷ lệ nhỏ và kiểm tra nguồn tải ảnh khi chưa chọn tệp. Đã kiểm tra trực tiếp bằng computer-use trên Chrome: chọn ảnh mẫu, đổi ảnh, tải tệp ảnh từ máy, phân tích và tải đủ ba PNG cùng CSV. Các tệp tải xuống đã được đọc lại và đối chiếu pixel/diện tích thành công. CLI đã kiểm tra ghi ba ảnh vào thư mục tiếng Việt và xử lý lỗi ghi mà không báo thành công giả. Môi trường chạy: Python 3.14.7, OpenCV 5.0.0, NumPy 2.5.4, Streamlit 1.67.0.
- Đã kiểm tra 10 ảnh thật trong bộ dữ liệu theo nguồn gốc nhóm cung cấp. Tất cả mask mới đúng kích thước, chỉ chứa 0/255 và có **0 pixel đốm ngoài lá**.
- Lá vàng đồng `normal/3.png`: tỷ lệ bị đánh dấu giảm từ **79,9452%** xuống **0,1195%**. Lá có đốm trên mặt dưới `spotted/hu1.png`: từ **76,8373%** xuống **4,2908%**, giữ các đốm cục bộ thay cho gần cả mặt lá.
- Còn vùng nhận nhầm: ví dụ `normal/1.png` vẫn có **0,5406%** bị đánh dấu sát chóp lá; `normal/3.png` còn các mảnh gân. Thuật toán không bảo đảm xóa hết gân/cuống hoặc bóng tối.
- Đã kiểm tra 30 lượt đổi sáng trên cùng 10 ảnh; đây là biến đổi mô phỏng sau tiền xử lý, không phải 30 ảnh chụp độc lập. Mask luôn trong lá nhưng diện tích đốm thay đổi theo sáng.
- Chưa có mask đốm gán nhãn chuẩn. Tỷ lệ diện tích, số thành phần và kiểm tra kỹ thuật **không phải** độ chính xác phân đoạn hay bằng chứng chẩn đoán bệnh. Cần gán nhãn và đo IoU/Dice để đánh giá tiếp.
- Phép lọc gân/biên có thể bỏ nhầm đốm nhỏ dài, đốm sát mép. Ngưỡng màu cần hiệu chỉnh khi đổi điều kiện chụp; lá đổi màu đồng đều và mô bệnh cùng màu lá có thể bị bỏ sót.

## 6. Kết quả rà soát và chạy trình diễn

- Không phát hiện lỗi làm sai hợp đồng `detect_spots(processed_bgr, leaf_mask) -> spot_mask` trong các kiểm thử hiện có; không thay đổi ngưỡng của detector trong lần rà soát này.
- Đã khắc phục CLI bỏ qua kết quả ghi ảnh: dùng `cv2.imencode` kết hợp ghi byte, hỗ trợ tên Unicode và bắt lỗi I/O.
- Đã sửa mã hóa stdout/stderr của CLI sang UTF-8, tránh lỗi `UnicodeEncodeError` khi in đường dẫn tiếng Việt trên console/pipe Windows mặc định cp1252; kiểm tra bằng tiến trình CLI thật với `PYTHONIOENCODING=cp1252`.
- Bổ sung giao diện tiếng Việt với ảnh mẫu, nút phân tích, tải kết quả và bảo vệ số liệu cũ khi đổi ảnh. Với số pixel đốm khác 0 nhưng tỷ lệ làm tròn về 0, hiển thị `<0.01%` để không nhầm với 0 tuyệt đối.
- Chạy lại công cụ đánh giá trên 10 ảnh và 30 biến thể sáng; kết quả detector không đổi so với bảng ở Chương 3, tất cả vẫn có 0 pixel đốm ngoài lá.
- Máy chủ Streamlit cục bộ ở `http://127.0.0.1:8501`. Lần kiểm tra trước bị dừng ở bước chuyển trang vì công cụ không xác định đủ chắc URL. Trong lần tiếp tục, đã kiểm tra được trực tiếp trên trang đang mở và hoàn tất các thao tác còn thiếu. Xem [biên bản kiểm tra giao diện](KIEM_TRA_GIAO_DIEN_NHUT.md); các tệp tải bằng trình duyệt nằm trong `results/nhut/ui_checks/`.
