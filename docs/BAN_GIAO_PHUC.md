# BIÊN BẢN BÀN GIAO KỸ THUẬT - PHẦN ĐẶNG HOÀNG PHÚC

**Dự án:** Phát hiện đốm bất thường trên lá sầu riêng bằng kỹ thuật xử lý ảnh  
**Người bàn giao:** **Đặng Hoàng Phúc** (Phụ trách Phân tích mức độ nghiêm trọng, Giao diện Web Streamlit, Kiểm thử ứng dụng & Chương 4)  
**Người nhận bàn giao:**  
1. **Lê Hữu Hào** (Nhóm trưởng - Tiền xử lý, Tách phiến lá, Tích hợp & Chương 1-2)  
2. **Trần Minh Nhựt** (Phụ trách Phát hiện đốm bất thường - `src/spot_detection.py` & Chương 3)  
**Thời gian bàn giao:** 12/11/2026  

---

## 1. DANH SÁCH CÁC SẢN PHẨM BÀN GIAO

| STT | Tên tệp / Sản phẩm | Vai trò & Mục đích sử dụng | Trạng thái kỹ thuật |
| :---: | :--- | :--- | :---: |
| 1 | `src/severity_analysis.py` | Hàm tính diện tích pixel lá/đốm, tỷ lệ % và xếp loại 4 cấp độ bệnh | **HOÀN THÀNH 100%** |
| 2 | `app.py` | Giao diện Web trực quan nền tảng Streamlit, 4 bước xem ảnh, tải mask & CSV | **HOÀN THÀNH 100%** |
| 3 | `docs/CHUONG_4_PHUC.md` | Báo cáo chi tiết Chương 4: Thực nghiệm, kiểm thử, phân tích và hướng phát triển | **HOÀN THÀNH 100%** |
| 4 | `docs/kiem_thu_phuc.csv` | Bảng kiểm thử thực nghiệm 10 ảnh mẫu chụp thật thông qua giao diện Web | **HOÀN THÀNH 100%** |
| 5 | `docs/HUONG_DAN_QUAY_VIDEO_DEMO.md` | Kịch bản chi tiết và lời thoại phục vụ quay video demo nộp bài | **HOÀN THÀNH 100%** |

---

## 2. QUY CÁCH CHỮ KÝ HÀM VÀ DỮ LIỆU ĐẦU RA

### 2.1. Hàm phân tích mức độ nghiêm trọng (`src/severity_analysis.py`)

Chữ ký hàm:
```python
def calculate_severity(leaf_mask: np.ndarray, spot_mask: np.ndarray) -> dict:
    ...
```

- **Quy cách đầu vào:**
  - `leaf_mask`: Mảng NumPy 2 chiều `(H, W)`, kiểu `uint8`. Pixel giá trị `255` là phiến lá, giá trị `0` là nền.
  - `spot_mask`: Mảng NumPy 2 chiều `(H, W)`, kiểu `uint8`. Pixel giá trị `255` là đốm bệnh, giá trị `0` là mô lá khỏe. Đảm bảo đốm nằm gọn bên trong lá (`cv2.bitwise_and(spot_mask, leaf_mask)`).
- **Quy cách từ điển trả về (`dict`):**
  - `"leaf_area_px"` (`int`): Tổng số pixel phiến lá ($A_{\text{lá}}$).
  - `"spot_area_px"` (`int`): Tổng số pixel đốm bệnh ($A_{\text{đốm}}$).
  - `"severity_percentage"` (`float`): Tỷ lệ tổn thương $S = (A_{\text{đốm}} / A_{\text{lá}}) \times 100$, làm tròn 2 chữ số thập phân.
  - `"grade"` (`str`): Cấp độ xếp loại (`Healthy (Normal)`, `Low / Early Stage`, `Moderate Stage`, `Severe Stage`).

### 2.2. Giao diện Web Streamlit (`app.py`)

- Tích hợp pipeline 4 bước: Tiền xử lý $\rightarrow$ Tách lá GrabCut $\rightarrow$ Phát hiện đốm HSV/LAB $\rightarrow$ Phân tích mức độ và khoanh viền đỏ.
- Cơ chế Cache gắn với SHA-256 mã nguồn của `src/spot_detection.py` (`DETECTOR_VERSION`), tránh việc sử dụng cache lỗi thời.
- Cơ chế chống hiển thị số liệu cũ: Xóa state phân tích khi đổi ảnh mẫu hoặc chuyển đổi nguồn tải file.
- Hỗ trợ xuất dữ liệu: Nút tải 3 file ảnh PNG (Mask lá, Mask đốm, Ảnh khoanh đỏ) và file CSV số liệu UTF-8 BOM.

---

## 3. HƯỚNG DẪN KHỞI CHẠY VÀ KIỂM THỬ

1. **Khởi chạy ứng dụng Web:**
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
   Ứng dụng sẽ mở tại `http://localhost:8501`.

2. **Chạy kiểm thử tự động luồng giao diện:**
   ```powershell
   .\.venv\Scripts\python.exe -m unittest tests/test_app.py -v
   ```

---

## 4. XÁC NHẬN BÀN GIAO

- **Người bàn giao:** Đặng Hoàng Phúc (Đã ký và hoàn tất)
- **Người nhận bàn giao:** Lê Hữu Hào, Trần Minh Nhựt (Đã đồng bộ vào `main.py` và kiểm thử luồng hoạt động)
