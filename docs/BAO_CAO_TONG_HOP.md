# BÁO CÁO TỔNG HỢP ĐỀ TÀI HỌC PHẦN XỬ LÝ ẢNH

## ĐỀ TÀI: PHÁT HIỆN ĐỐM BẤT THƯỜNG TRÊN LÁ SẦU RIÊNG VÀ PHÂN TÍCH MỨC ĐỘ BỆNH BẰNG KỸ THUẬT THỊ GIÁC MÁY TÍNH
**Học phần:** Xử lý ảnh  
**Ngôn ngữ & Nền tảng:** Python 3.10+, OpenCV, Streamlit  

---

### THÀNH VIÊN NHÓM VÀ BẢNG PHÂN CÔNG ĐÓNG GÓP

| STT | Họ và tên sinh viên | Vai trò phụ trách | Nhiệm vụ kỹ thuật chính | Báo cáo & Tỷ lệ đóng góp |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Lê Hữu Hào** | Nhóm trưởng<br>Tiền xử lý & Tích hợp | - Quản lý cấu trúc dự án, thu thập 10 ảnh thật (`data/`).<br>- Xây dựng hàm tiền xử lý `src/preprocessing.py` (resize, GaussianBlur, CLAHE).<br>- Tách phiến lá `src/leaf_segmentation.py` (GrabCut, Otsu, Sanity Checks).<br>- Tích hợp CLI [main.py](file:///d:/XLA/durian-leaf-spots-main/main.py) và điều phối toàn hệ thống. | Chương 1 & Chương 2<br>Tổng hợp báo cáo<br>**34%** |
| 2 | **Trần Minh Nhựt** | Thuật toán phát hiện đốm | - Thiết lập không gian màu HSV và LAB, ước lượng màu tham chiếu phiến lá.<br>- Phát hiện đốm bất thường `src/spot_detection.py`.<br>- Lọc nhiễu Morphology, loại trừ gân lá, cuống hẹp và bóng mép lá.<br>- Khoanh viền đỏ và kiểm thử độ bền ánh sáng `scripts/evaluate_spots.py`. | Chương 3<br>Minh chứng & So sánh<br>**33%** |
| 3 | **Đặng Hoàng Phúc** | Phân tích mức độ & Giao diện | - Viết module tính diện tích và tỷ lệ tổn thương `src/severity_analysis.py`.<br>- Thiết lập thang phân cấp 4 mức cảnh báo (Healthy, Low, Moderate, Severe).<br>- Xây dựng giao diện ứng dụng Web [app.py](file:///d:/XLA/durian-leaf-spots-main/app.py) trên Streamlit.<br>- Thực nghiệm kiểm thử 10 ảnh mẫu và lập kịch bản video demo. | Chương 4<br>Kiểm thử & Giao diện<br>**33%** |

---

# MỤC LỤC
1. [Chương 1: Tổng quan bài toán, mục tiêu và phạm vi nghiên cứu](#chương-1-tổng-quan-bài-toán-mục-tiêu-và-phạm-vi-nghiên-cứu)
2. [Chương 2: Phương pháp tiền xử lý và phân đoạn tách phiến lá sầu riêng](#chương-2-phương-pháp-tiền-xử-lý-và-phân-đoạn-tách-phiến-lá-sầu-riêng)
3. [Chương 3: Phương pháp và thuật toán phát hiện đốm bất thường](#chương-3-phương-pháp-và-thuật-toán-phát-hiện-đốm-bất-thường)
4. [Chương 4: Đánh giá thực nghiệm, kiểm thử và hướng phát triển đề tài](#chương-4-đánh-giá-thực-nghiệm-kiểm-thử-và-hướng-phát-triển-đề-tài)
5. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

# CHƯƠNG 1: TỔNG QUAN BÀI TOÁN, MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU
*(Biên soạn: Lê Hữu Hào)*

## 1.1. Đặt vấn đề
Cây sầu riêng (*Durio zibethinus*) là cây ăn trái chủ lực mang lại giá trị kinh tế rất lớn tại Việt Nam. Tuy nhiên, cây sầu riêng đặc biệt nhạy cảm với các bệnh nấm và vi khuẩn gây đốm hoại tử trên lá như đốm lá rong (*Algal leaf spot*), thán thư (*Colletotrichum spp.*), đốm mắt cua (*Phomopsis durionis*) và cháy lá chết ngọn (*Rhizoctonia solani*). 

Việc thăm vườn và phát hiện bệnh bằng mắt thường phụ thuộc cảm tính, khó lượng hóa chính xác tỷ lệ tổn thương để quyết định liều lượng thuốc bảo vệ thực vật phù hợp. Đề tài áp dụng kỹ thuật Thị giác máy tính (Computer Vision) và Xử lý ảnh số truyền thống để tự động tách phiến lá, phát hiện các vùng đốm hoại tử bất thường và tính toán tỷ lệ diện tích tổn thương chính xác.

## 1.2. Mục tiêu nghiên cứu
1. Chuẩn hóa kích thước ảnh đầu vào, bảo toàn tỷ lệ khung hình (aspect ratio), khử nhiễu cảm biến nhẹ và tăng cường độ tương phản thích nghi bằng CLAHE.
2. Tách phiến lá chính xác khỏi hậu cảnh vườn hoặc bàn chụp, lấp đầy các lỗ thủng bên trong phiến lá do đốm hoại tử gây ra.
3. Phát hiện chính xác các đốm màu nâu, vàng và đen bệnh lý bằng sự phối hợp giữa hai không gian màu HSV và LAB, loại bỏ nhiễu gân lá và bóng mép lá.
4. Định lượng diện tích tổn thương $S = (A_{\text{đốm}} / A_{\text{lá}}) \times 100\%$ và phân cấp cảnh báo tự động.
5. Triển khai sản phẩm hoàn chỉnh trên giao diện dòng lệnh (CLI) và giao diện Web tương tác (Streamlit).

## 1.3. Phạm vi và Giới hạn kỹ thuật
- **Hỗ trợ tối ưu:** Ảnh chụp một phiến lá sầu riêng làm chủ đạo ở trung tâm khung hình, trên nền đơn sắc phòng thí nghiệm hoặc nền đất vườn thoáng sáng.
- **Giới hạn:** Chưa hỗ trợ tự động bóc tách từng lá trong chùm nhiều lá chồng lấn đan xen (người dùng có thể định vị vùng quan tâm bằng cờ `--roi x y w h`).

---

# CHƯƠNG 2: PHƯƠNG PHÁP TIỀN XỬ LÝ VÀ PHÂN ĐOẠN TÁCH PHIẾN LÁ
*(Biên soạn: Lê Hữu Hào)*

## 2.1. Tiền xử lý ảnh (`src/preprocessing.py`)
1. **Đọc ảnh an toàn (`load_image`):** Sử dụng `cv2.imdecode` kết hợp `np.fromfile` để hỗ trợ tuyệt đối đường dẫn tệp chứa ký tự tiếng Việt có dấu trên hệ điều hành Windows.
2. **Resize giữ tỷ lệ (Aspect Ratio):** Thu nhỏ cạnh dài nhất về ngưỡng tối đa 1024 pixel bằng nội suy diện tích `cv2.INTER_AREA`, tránh méo mó hình học của phiến lá.
3. **Giảm nhiễu nhẹ:** Bộ lọc làm mịn Gaussian kernel $3 \times 3$, $\sigma = 0.8$ nhằm làm mịn hạt nhiễu cảm biến mà không làm mờ các chấm bệnh nhỏ li ti.
4. **Cân bằng sáng cục bộ thích nghi (CLAHE):** Chuyển sang không gian màu $L^*a^*b^*$, áp dụng thuật toán CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`) trên kênh độ sáng $L^*$, sau đó chuyển ngược về BGR.

## 2.2. Phân đoạn tách phiến lá (`src/leaf_segmentation.py`)
1. **Khởi tạo vùng ROI:** Tự động tạo hình chữ nhật trung tâm chiếm $85\%$ khung hình để phân định vùng foreground tiềm năng.
2. **Phân đoạn bán tự động GrabCut:** Áp dụng thuật toán GrabCut với 5 vòng lặp tối ưu hóa đồ thị (Graph Cut) để tách thô biên giới phiến lá.
3. **Phân đoạn ngưỡng cục bộ Otsu:** Kết hợp với mặt nạ màu xanh lục trên kênh $a^*$ và chỉ số thực vật ExG ($2G - R - B$) để loại bỏ các vùng đất hoặc bóng đổ còn sót lại.
4. **Lấp đầy lỗ thủng nội tại (Hole Filling):** Các vết đốm hoại tử màu nâu đen thường bị thuật toán gán nhầm thành nền (background). Module tìm tất cả đường bao bên trong phiến lá và lấp đầy bằng cờ `cv2.FILLED`, đảm bảo $M_{\text{lá}}$ là một khối đặc trơn tru.
5. **Cơ chế Sanity Checks:** Kiểm tra diện tích lá phải nằm trong khoảng $5\% \le A_{\text{lá}} \le 85\%$ diện tích khung hình và tỷ lệ tiếp xúc 4 mép ảnh không vượt quá $30\%$, nếu vi phạm sẽ ném ngoại lệ `LeafSegmentationError`.

---

# CHƯƠNG 3: PHƯƠNG PHÁP VÀ THUẬT TOÁN PHÁT HIỆN ĐỐM BẤT THƯỜNG
*(Biên soạn: Trần Minh Nhựt)*

## 3.1. Hợp đồng dữ liệu đầu vào và đầu ra
Hàm thực thi có chữ ký chuẩn:
```python
def detect_spots(processed_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
```
- Đầu vào: `processed_bgr` $(H, W, 3)$ và `leaf_mask` $(H, W)$ chuẩn `uint8`.
- Đầu ra: Duy nhất một mảng nhị phân `spot_mask` $(H, W)$, với quy ước $255$ là đốm và $0$ là vùng lá bình thường. Luôn thỏa mãn ràng buộc logic: $\forall (x,y), M_{\text{lá}}(x,y) = 0 \Rightarrow M_{\text{đốm}}(x,y) = 0$.

## 3.2. Thuật toán phát hiện đốm phối hợp HSV & LAB
1. **Không gian màu HSV:**
   - Đốm màu nâu/đỏ hoại tử: $H \in [0, 20] \cup [160, 179]$, $S \in [40, 255]$, $V \in [30, 200]$.
   - Đốm màu vàng/cháy lá: $H \in [20, 38]$, $S \in [60, 255]$, $V \in [60, 240]$.
   - Vết hoại tử tối màu/thâm đen: $V < 45$, kết hợp kiểm tra độ lệch màu so với màu xanh chuẩn của lá.
2. **Không gian màu LAB:**
   - Kênh $b^*$ đặc trưng cho sắc tố vàng ($b^* > 140$).
   - Kênh $a^*$ phát hiện sắc tố đỏ/nâu hoại tử ($a^* > 130$).
3. **Lọc hình thái học & Khử nhiễu hình học:**
   - Áp dụng Morphological Opening với kernel elip $3 \times 3$ để khử chấm nhiễu đơn lẻ.
   - Morphological Closing với kernel elip $5 \times 5$ để liền mạch các mảng đốm rải rác.
   - Phân tích thành phần liên thông (`cv2.connectedComponentsWithStats`): Loại bỏ các vùng có diện tích $< 15$ pixel.
   - Loại trừ gân lá bằng tỷ lệ trục dài/trục ngắn (Aspect Ratio $> 4.0$) và hướng dọc thân lá.
   - Khử nhiễu bóng mép lá: Phép xói mòn (Erosion) viền mép lá 5 pixel, chỉ giữ lại đốm mép nếu thỏa mãn đồng thời tiêu chuẩn màu nâu hoại tử rõ nét.
4. **Vẽ đường viền hiển thị:** Tìm đường viền `cv2.findContours` từ mask đốm đã lọc và vẽ viền đỏ `(0, 0, 255)` độ dày 2 pixel lên ảnh tiền xử lý.

---

# CHƯƠNG 4: ĐÁNH GIÁ THỰC NGHIỆM, KIỂM THỬ VÀ HƯỚNG PHÁT TRIỂN
*(Biên soạn: Đặng Hoàng Phúc)*

## 4.1. Định lượng mức độ nghiêm trọng & Phân cấp bệnh
- Công thức tính diện tích pixel: $A_{\text{lá}} = \text{countNonZero}(M_{\text{lá}})$, $A_{\text{đốm}} = \text{countNonZero}(M_{\text{đốm}})$.
- Tỷ lệ tổn thương: $S = (A_{\text{đốm}} / A_{\text{lá}}) \times 100\%$.
- Thang phân cấp 4 mức cảnh báo:
  - $S = 0.0\%$: **Healthy (Bình thường)**.
  - $0\% < S < 5.0\%$: **Low / Early Stage (Nhẹ)**.
  - $5.0\% \le S < 20.0\%$: **Moderate Stage (Trung bình)**.
  - $S \ge 20.0\%$: **Severe Stage (Nặng)**.

## 4.2. Giao diện người dùng Streamlit (`app.py`)
- Thiết kế 4 bước xem ảnh: Ảnh tiền xử lý $\rightarrow$ Mask phiến lá $\rightarrow$ Mask vết đốm $\rightarrow$ Ảnh đốm khoanh viền đỏ.
- Cơ chế bảo vệ dữ liệu: Gắn mã hash SHA-256 thuật toán vào cache; xóa bỏ kết quả cũ ngay khi chuyển ảnh hoặc đổi nguồn tải file để loại bỏ nhầm lẫn số liệu.
- Hỗ trợ xuất dữ liệu: Nút tải 3 ảnh PNG chuẩn phân giải và bảng số liệu CSV UTF-8.

## 4.3. Bảng tổng hợp kết quả thực nghiệm trên 10 ảnh mẫu chụp thật

| STT | Tên ảnh | Nhãn thực tế | Diện tích lá (px) | Diện tích đốm (px) | Tỷ lệ tổn thương | Xếp loại cấp độ | Đánh giá nhận diện |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `1.png` | Bình thường | 403,816 | 2,183 | $0.54\%$ | Nhẹ | Bắt vết viền mép lá nhỏ |
| 2 | `2.png` | Bình thường | 474,320 | 20 | $<0.01\%$ | Bình thường | Nhiễu bóng mờ 20 pixel |
| 3 | `3.png` | Bình thường | 349,807 | 418 | $0.12\%$ | Nhẹ | Khắc phục ngoạn mục (từ 79.95% về 0.12%) |
| 4 | `4.png` | Bình thường | 467,810 | 0 | $0.00\%$ | Bình thường | Chuẩn xác tuyệt đối (0 pixel đốm) |
| 5 | `5.png` | Bình thường | 303,837 | 0 | $0.00\%$ | Bình thường | Chuẩn xác tuyệt đối (0 pixel đốm) |
| 6 | `hu1.png` | Có đốm | 377,274 | 16,188 | $4.29\%$ | Nhẹ | Bắt chính xác vùng cháy bìa lá |
| 7 | `hu2.png` | Có đốm | 558,256 | 171,535 | $30.73\%$ | Nặng | Đốm cháy loang rộng > 30% |
| 8 | `hu3.png` | Có đốm | 410,643 | 10,239 | $2.49\%$ | Nhẹ | Bắt trúng đốm chấm hoại tử |
| 9 | `hu4.png` | Có đốm | 336,574 | 38,720 | $11.50\%$ | Trung bình | Phát hiện cụm đốm phân tán |
| 10 | `hu5.png` | Có đốm | 636,407 | 12,752 | $2.00\%$ | Nhẹ | Bắt đúng đốm góc đuôi lá |

## 4.4. Đánh giá ưu điểm, hạn chế và hướng phát triển
- **Ưu điểm:** Tốc độ xử lý cực nhanh ($< 1$ giây trên CPU), thuật toán tường minh dễ giải thích, giao diện trực quan, tính toán diện tích khách quan không phụ thuộc cảm tính.
- **Hạn chế:** Nhạy cảm với ảnh nhiều lá đan xen phức tạp hoặc mặt dưới lá màu đồng bị phơi sáng gắt; hiện chưa có Ground-Truth Mask chuẩn pixel để đo chỉ số IoU/Dice.
- **Hướng phát triển:** Xây dựng tập dữ liệu lớn (>500 ảnh) có gán nhãn Ground-Truth; ứng dụng Deep Learning (YOLOv8-Seg) để tách lá chùm; phát triển ứng dụng di động cho nông dân chụp trực tiếp tại vườn.

---

# TÀI LIỆU THAM KHẢO
1. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
2. Rother, C., Kolmogorov, V., & Blake, A. (2004). *"GrabCut": interactive foreground extraction using iterated graph cuts*. ACM Transactions on Graphics (TOG), 23(3), 309-314.
3. Otsu, N. (1979). *A threshold selection method from gray-level histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
5. Cục Trồng trọt - Bộ Nông nghiệp & PTNT. *Quy trình kỹ thuật quản lý sinh vật gây hại trên cây sầu riêng*.
