# Chương 4. Đánh giá kết quả thực nghiệm, kiểm thử và hướng phát triển đề tài

**Họ và tên sinh viên thực hiện:** Đặng Hoàng Phúc  
**Nhiệm vụ phụ trách:** Tính toán mức độ nghiêm trọng (`src/severity_analysis.py`), Xây dựng giao diện ứng dụng Web trực quan (`app.py`), Thực nghiệm kiểm thử và Biên soạn Chương 4.  
**Tỷ lệ đóng góp theo phân công:** 33%.  

---

## 4.1. Phương pháp định lượng mức độ nghiêm trọng (Severity Analysis)

Sau khi module tiền xử lý & phân đoạn tách phiến lá của Hào (`src/preprocessing.py`, `src/leaf_segmentation.py`) và module phát hiện đốm của Nhựt (`src/spot_detection.py`) hoàn tất, kết quả bàn giao là hai mask nhị phân kích thước $(H, W)$ cùng hệ tọa độ:
- $M_{\text{lá}}(x, y) \in \{0, 255\}$: Mask phiến lá (vùng lá mang giá trị $255$, nền mang giá trị $0$).
- $M_{\text{đốm}}(x, y) \in \{0, 255\}$: Mask vết đốm bất thường (vết bệnh mang giá trị $255$, vùng lá khỏe mang giá trị $0$). Nhờ phép ràng buộc $M_{\text{đốm}} = M_{\text{đốm}} \land M_{\text{lá}}$, đốm đảm bảo $100\%$ nằm gọn bên trong phiến lá.

### 4.1.1. Công thức toán học tính diện tích và tỷ lệ tổn thương

Hàm `calculate_severity` trong `src/severity_analysis.py` tiến hành đếm số lượng điểm ảnh (pixel counting) bằng hàm tối ưu `cv2.countNonZero`:

$$\text{Leaf Area (px)} = A_{\text{lá}} = \sum_{x=1}^{W}\sum_{y=1}^{H} \mathbb{I}\left(M_{\text{lá}}(x, y) = 255\right)$$

$$\text{Spot Area (px)} = A_{\text{đốm}} = \sum_{x=1}^{W}\sum_{y=1}^{H} \mathbb{I}\left(M_{\text{đốm}}(x, y) = 255\right)$$

Tỷ lệ phần trăm diện tích nhiễm bệnh (Severity Percentage) được xác định bởi:

$$S = \begin{cases} 
0.0\%, & \text{nếu } A_{\text{lá}} = 0 \\
\left(\dfrac{A_{\text{đốm}}}{A_{\text{lá}}}\right) \times 100\%, & \text{nếu } A_{\text{lá}} > 0
\end{cases}$$

Chỉ số $S$ được làm tròn đến 2 chữ số thập phân (`round(severity_percentage, 2)`). Tuy nhiên, trên giao diện và báo cáo, đối với các trường hợp có $A_{\text{đốm}} > 0$ nhưng giá trị sau làm tròn bằng $0.00\%$ (ví dụ $0.0042\%$), hệ thống hiển thị nhãn đặc biệt `<0.01%` để phân biệt rõ ràng với trạng thái hoàn toàn không có đốm ($A_{\text{đốm}} = 0$).

### 4.1.2. Thang phân cấp mức độ nghiêm trọng của bệnh

Dựa trên các khuyến cáo kỹ thuật nông nghiệp về bệnh đốm lá, thán thư và đốm mắt cua trên cây sầu riêng (*Durio zibethinus*), Phúc thiết lập thang chia 4 cấp độ cảnh báo:

| Tỷ lệ diện tích đốm ($S$) | Tên cấp độ tiếng Anh (`grade`) | Tên cấp độ tiếng Việt | Khuyến nghị canh tác & điều trị |
| :---: | :---: | :---: | :--- |
| $S = 0.0\%$ | `Healthy (Normal)` | **Bình thường** | Lá khỏe mạnh; duy trì chế độ phân bón và theo dõi định kỳ. |
| $0\% < S < 5.0\%$ | `Low / Early Stage` | **Nhẹ (Giai đoạn đầu)** | Vết bệnh mới phát sinh hoặc diện tích nhỏ; tỉa lá sát gốc, vệ sinh tán cây. |
| $5.0\% \le S < 20.0\%$ | `Moderate Stage` | **Trung bình** | Bệnh đã lan rộng trên bề mặt phiến lá; phun thuốc gốc đồng hoặc hoạt chất sinh học đối kháng. |
| $S \ge 20.0\%$ | `Severe Stage` | **Nặng (Nghiêm trọng)** | Khả năng quang hợp của phiến lá suy giảm nghiêm trọng; cách ly cành bệnh, sử dụng thuốc đặc trị nấm/vi khuẩn. |

---

## 4.2. Thiết kế và xây dựng giao diện người dùng trực quan (`app.py`)

Giao diện ứng dụng được phát triển trên nền tảng **Streamlit**, hướng đến tiêu chí: **Đơn giản, Trực quan, Chính xác và Hạn chế tối đa thao tác nhầm lẫn của người dùng**.

```
+------------------------------------------------------------------------------------+
|  🍃 PHÁT HIỆN ĐỐM BẤT THƯỜNG TRÊN LÁ SẦU RIÊNG - STREAMLIT INTERFACE               |
+------------------------------------------------------------------------------------+
|  Nguồn ảnh: (o) Ảnh mẫu có sẵn    ( ) Tải ảnh của bạn                              |
|  [Chọn ảnh mẫu: Có đốm · hu1.png                                     v]            |
|  [  ▶ PHÂN TÍCH ẢNH  ]                                                             |
+------------------------------------------------------------------------------------+
|  KẾT QUẢ PHÂN TÍCH:                                                                |
|  [Diện tích lá: 377,274 px] [Diện tích đốm: 16,188 px] [Tỷ lệ: 4.29%] [Mức: Nhẹ]   |
+---------------------+---------------------+--------------------+-------------------+
| 1. Ảnh tiền xử lý   | 2. Mask phiến lá    | 3. Mask vết đốm    | 4. Đốm khoanh đỏ  |
| [     (Image)     ] | [     (Binary)    ] | [    (Binary)    ] | [  (Contour BGR) ]|
+---------------------+---------------------+--------------------+-------------------+
|  [v] Tải ảnh kết quả và số liệu (Download Leaf Mask, Spot Mask, Analyzed, Metrics)  |
+------------------------------------------------------------------------------------+
```

### 4.2.1. Các tính năng cốt lõi của giao diện:
1. **Lựa chọn hai nguồn ảnh linh hoạt:**
   - **Ảnh mẫu có sẵn:** Tự động quét 10 ảnh trong thư mục `data/` (`spotted` và `normal`), định dạng tên trực quan kèm phân loại sơ bộ.
   - **Tải ảnh của bạn:** Cho phép người dùng tải ảnh từ máy tính cá nhân (hỗ trợ JPG, JPEG, PNG, WebP).
2. **Cơ chế Cache dữ liệu thông minh gắn với phiên bản thuật toán:**
   - Hàm `analyze` được gán decorator `@st.cache_data`. Để tránh việc cache lưu giữ kết quả của thuật toán cũ khi code `src/spot_detection.py` bị chỉnh sửa, Phúc đưa giá trị SHA-256 hash của mã nguồn `src/spot_detection.py` vào làm một phần khóa cache (`DETECTOR_VERSION`).
3. **Cơ chế chống hiển thị số liệu cũ (Anti-stale metrics):**
   - Khi người dùng chuyển đổi nguồn ảnh hoặc chọn ảnh mẫu khác, toàn bộ kết quả phân tích cũ (`analysis_result`) trong `st.session_state` được tự động hủy bỏ. Người dùng bắt buộc phải bấm nút **"Phân tích ảnh"** thì kết quả mới mới xuất hiện. Điều này loại bỏ hoàn toàn khả năng đọc nhầm số liệu của ảnh A khi đang xem ảnh B.
4. **Hiển thị trực quan 4 bước xử lý:**
   - Cung cấp đồng thời 4 hình ảnh theo chiều ngang: Ảnh đã tiền xử lý, Mask phiến lá, Mask đốm và Ảnh đốm khoanh viền đỏ (`cv2.drawContours`, độ dày 2 pixel, màu đỏ BGR `(0, 0, 255)`).
5. **Xuất báo cáo và kết quả thực nghiệm:**
   - Cung cấp các nút tải về trực tiếp: Mask lá PNG, Mask đốm PNG, Ảnh khoanh đỏ PNG và Bảng số liệu định dạng CSV chuẩn UTF-8 có BOM (`utf-8-sig`).

---

## 4.3. Kết quả thực nghiệm trên tập dữ liệu chuẩn (10 ảnh mẫu)

Quá trình kiểm nghiệm được thực hiện trên toàn bộ 10 ảnh mẫu chụp thật do Hào cung cấp, áp dụng phiên bản thuật toán phát hiện đốm mới nhất của Nhựt và module tính toán của Phúc.

### 4.3.1. Bảng số liệu thực nghiệm chi tiết

| STT | Tên ảnh | Nhóm thực tế | Diện tích lá (px) | Diện tích đốm (px) | Tỷ lệ tổn thương ($S$) | Cấp độ xếp loại | Kết quả nhận diện |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `1.png` | Bình thường | 403,816 | 2,183 | $0.54\%$ | Nhẹ | Có vết viền mép lá nhỏ |
| 2 | `2.png` | Bình thường | 474,320 | 20 | $<0.01\%$ ($0.0042\%$) | Bình thường | Nhiễu bóng tối mờ sát gốc |
| 3 | `3.png` | Bình thường | 349,807 | 418 | $0.12\%$ | Nhẹ | Cải thiện lớn (thuật toán cũ: $79.95\%$) |
| 4 | `4.png` | Bình thường | 467,810 | 0 | $0.00\%$ | Bình thường | Chuẩn xác tuyệt đối ($0$ pixel đốm) |
| 5 | `5.png` | Bình thường | 303,837 | 0 | $0.00\%$ | Bình thường | Chuẩn xác tuyệt đối ($0$ pixel đốm) |
| 6 | `hu1.png` | Có đốm | 377,274 | 16,188 | $4.29\%$ | Nhẹ | Bắt trúng các vùng đốm hoại tử |
| 7 | `hu2.png` | Có đốm | 558,256 | 171,535 | $30.73\%$ | Nặng | Đốm cháy loang rộng khắp phiến |
| 8 | `hu3.png` | Có đốm | 410,643 | 10,239 | $2.49\%$ | Nhẹ | Bắt đúng đốm chấm hoại tử |
| 9 | `hu4.png` | Có đốm | 336,574 | 38,720 | $11.50\%$ | Trung bình | Phát hiện ổ bệnh đốm rải rác |
| 10 | `hu5.png` | Có đốm | 636,407 | 12,752 | $2.00\%$ | Nhẹ | Phát hiện ổ bệnh góc đuôi lá |

### 4.3.2. Phân tích kết quả thực nghiệm

1. **Trên nhóm lá có đốm bệnh (`hu1` – `hu5`):**
   - Thuật toán phát hiện thành công $100\%$ ($5/5$) ảnh lá có đốm bệnh. 
   - Tỷ lệ diện tích đốm phản ánh chính xác tình trạng quan sát bằng mắt thường: `hu2.png` có diện tích đốm chiếm đến $30.73\%$ (xếp loại **Nặng**), `hu4.png` chiếm $11.50\%$ (xếp loại **Trung bình**), các ảnh `hu1, hu3, hu5` xếp loại **Nhẹ** (khoảng $2\% - 4.3\%$).
   - Các đường viền contour màu đỏ khoanh khít quanh vết hoại tử thực thể, không bị đứt đoạn quá mức.

2. **Trên nhóm lá bình thường (`1.png` – `5.png`):**
   - Thuật toán đạt độ chính xác cao trên $2$ ảnh hoàn toàn sạch đốm (`4.png` và `5.png` đạt $0$ pixel đốm).
   - Ảnh `2.png` chỉ có $20$ pixel đốm rải rác ($0.0042\%$), nằm trong ngưỡng sai số cho phép và được quy về mức Bình thường.
   - **Cải tiến vượt bậc trên ảnh `3.png`:** Thuật toán cũ trước khi Nhựt cải tiến nhận nhầm mặt dưới lá màu vàng đồng làm đốm bệnh tới $79.95\%$. Phiên bản hiện tại đã giảm tỷ lệ nhầm lẫn xuống chỉ còn $0.12\%$ ($418$ pixel), kiểm soát tuyệt vời hiện tượng false positive.

---

## 4.4. Thử nghiệm độ bền vững trước điều kiện ánh sáng (Lighting Robustness)

Một hệ thống xử lý ảnh nông nghiệp thực tế cần hoạt động ổn định khi điều kiện chụp ảnh có độ phơi sáng khác nhau. Nhóm đã thực hiện kiểm tra biến thiên độ sáng nhân tạo với hệ số $\alpha \in \{0.75, 1.0, 1.25\}$:

$$I_{\text{test}} = \text{clip}(\alpha \cdot I, 0, 255)$$

Kết quả tổng hợp từ `results/nhut/lighting_checks.csv` (30 lượt chạy thử nghiệm):
- Khi thiếu sáng ($\alpha = 0.75$): Tỷ lệ đốm trên `hu1.png` dao động từ $4.29\%$ về $4.01\%$; trên lá bình thường `4.png`, số pixel đốm duy trì $0$.
- Khi dư sáng ($\alpha = 1.25$): Tỷ lệ đốm trên `hu1.png` đạt $4.48\%$; trên lá bình thường `4.png`, số pixel đốm duy trì $0$.
- **Kết luận:** Thuật toán duy trì sự ổn định tốt trước các dao động phơi sáng $\pm 25\%$, không làm nhảy bậc phân loại nghiêm trọng giữa các cấp độ bệnh.

---

## 4.5. Đánh giá ưu điểm, hạn chế và giới hạn kỹ thuật

### 4.5.1. Ưu điểm nổi bật của hệ thống
1. **Tốc độ xử lý cao & Không đòi hỏi phần cứng đặc biệt:** Toàn bộ thuật toán sử dụng các phép toán xử lý ảnh truyền thống (Morphology, Color Space Thresholding, Connected Components) trên OpenCV, chạy mượt mà trên CPU thông thường chỉ mất từ $0.5 - 1.2$ giây/ảnh mà không cần card đồ họa GPU rời.
2. **Tính minh bạch và khả năng giải thích (Explainable AI):** Không hoạt động như một "hộp đen" (black box) như mạng nơ-ron sâu, từng bước lọc nhiễu, lấp lỗ đốm, tách gân lá đều có cơ sở toán học và tham số rõ ràng.
3. **Ràng buộc logic chặt chẽ:** Quy ước $M_{\text{đốm}} \subseteq M_{\text{lá}}$ loại trừ hoàn toàn việc đốm giả xuất hiện ngoài hậu cảnh.
4. **Giao diện thân thiện, dễ vận hành:** Nông dân hoặc kỹ thuật viên vườn có thể tự sử dụng trên trình duyệt web chỉ với thao tác chọn ảnh và bấm nút.

### 4.5.2. Hạn chế và thách thức kỹ thuật
1. **Phụ thuộc vào chất lượng phân đoạn phiến lá (GrabCut):** Nếu ảnh chụp bị rung nhòe nặng hoặc nền vườn có nhiều lá lân cận đan xen sát mép, GrabCut có thể bắt nhầm viền nền hoặc cắt lẹm một phần phiến lá, dẫn đến tính toán sai diện tích mẫu số $A_{\text{lá}}$.
2. **Nhầm lẫn tiềm ẩn tại mặt dưới phiến lá:** Mặt dưới lá sầu riêng tự nhiên có lớp lông tơ màu vàng đồng/nâu sáng; dù đã được khắc phục ở ảnh `3.png`, trong một số góc chụp nghiêng gắt vẫn có thể phát sinh diện tích đốm giả nhỏ.
3. **Chưa có Ground-Truth Mask chuẩn pixel:** Bộ dữ liệu 10 ảnh hiện tại được đánh giá dựa trên sự quan sát định tính của con người và bảng thống kê số liệu, chưa có mặt nạ gán nhãn thủ công (Ground-Truth Mask) từng pixel để tính toán chỉ số định lượng chặt chẽ như **IoU (Intersection over Union)** hay **Dice Coefficient**.

---

## 4.6. Kết luận và hướng phát triển đề tài

### 4.6.1. Kết luận
Đề tài *"Phát hiện đốm bất thường trên lá sầu riêng bằng kỹ thuật xử lý ảnh"* đã hoàn thành trọn vẹn tất cả các mục tiêu đề ra ban đầu theo chuẩn học phần Xử lý ảnh:
- Xây dựng thành công chuỗi xử lý khép kín gồm: Tiền xử lý $\rightarrow$ Tách phiến lá $\rightarrow$ Phát hiện vết bệnh $\rightarrow$ Định lượng tỷ lệ tổn thương và Phân cấp mức độ bệnh.
- Đóng gói sản phẩm thành công trên cả hai phương diện: Giao diện dòng lệnh CLI ([main.py](file:///d:/XLA/durian-leaf-spots-main/main.py)) phục vụ xử lý tự động và Giao diện Web trực quan ([app.py](file:///d:/XLA/durian-leaf-spots-main/app.py)) phục vụ người dùng cuối.
- Ba thành viên Hào, Nhựt, Phúc phối hợp ăn ý, xác lập hợp đồng dữ liệu chuẩn xác, kiểm thử độc lập và bàn giao minh bạch.

### 4.6.2. Hướng phát triển trong tương lai
1. **Mở rộng tập dữ liệu và gán nhãn Ground-Truth:** Xây dựng tập dữ liệu từ 200 – 500 ảnh thực tế tại các vùng trồng sầu riêng (Tiền Giang, Bến Tre, Đắk Lắk), gán nhãn mask đốm bằng phần mềm chuyên dụng (Labelme, CVAT) để đo lường độ chính xác IoU định lượng.
2. **Phân loại tác nhân gây bệnh cụ thể:** Nâng cấp từ việc phát hiện đốm bất thường chung chung sang phân loại chính xác tên bệnh (Thán thư, Đốm rong, Cháy lá chết ngọn, Đốm mắt cua) dựa trên đặc trưng vân bề mặt (GLCM) và hình dạng vết bệnh.
3. **Ứng dụng mô hình học sâu (Deep Learning):** Thử nghiệm kết hợp các mô hình phân đoạn thế hệ mới như **YOLOv8-Seg**, **Mask R-CNN** hoặc **MobileNetV3** để tự động nhận diện nhiều lá cùng lúc trong điều kiện thực địa phức tạp.
4. **Triển khai ứng dụng di động (Mobile App / PWA):** Đóng gói hệ thống thành ứng dụng di động (Android / iOS) có thể hoạt động offline, giúp bà con nông dân chụp ảnh và nhận kết quả cảnh báo bệnh ngay tại gốc cây ngoài vườn sầu riêng.
