# KỊCH BẢN VÀ HƯỚNG DẪN QUAY VIDEO DEMO SẢN PHẨM

**Đề tài:** Phát hiện đốm bất thường trên lá sầu riêng bằng kỹ thuật xử lý ảnh  
**Thời lượng dự kiến:** 3 phút – 4 phút  
**Công cụ khuyến nghị để quay màn hình:** OBS Studio, Windows Xbox Game Bar (`Windows + Alt + R`), hoặc Camtasia / CapCut PC.  
**Chuẩn bị trước khi quay:**
1. Khởi động sẵn ứng dụng bằng lệnh:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
2. Mở trình duyệt tại `http://localhost:8501`, phóng to toàn màn hình (`F11` hoặc tối đa hóa cửa sổ).
3. Đảm bảo âm thanh micro rõ ràng, không bị rè hoặc lẫn tạp âm nền.

---

## KỊCH BẢN THUYẾT MINH CHI TIẾT TỪNG GIÂY

### ⏱️ Phần 1: Giới thiệu đề tài và thành viên (00:00 – 00:30)
- **Hành động trên màn hình:** Để màn hình trang chủ giao diện Streamlit, rê chuột chỉ vào tiêu đề và thanh Sidebar giới thiệu thông tin nhóm.
- **Lời thoại thuyết minh:**
  > *"Kính chào Thầy/Cô và các bạn! Hôm nay, nhóm chúng em gồm 3 thành viên: Lê Hữu Hào (Nhóm trưởng phụ trách Tiền xử lý, Tách phiến lá và Tích hợp), Trần Minh Nhựt (phụ trách Thuật toán phát hiện đốm bất thường) và Đặng Hoàng Phúc (phụ trách Phân tích mức độ nghiêm trọng và Xây dựng giao diện ứng dụng). Nhóm xin phép được demo chương trình: Hệ thống phát hiện đốm bất thường trên lá sầu riêng và định lượng diện tích tổn thương sử dụng OpenCV và Streamlit."*

---

### ⏱️ Phần 2: Demo tính năng phân tích ảnh mẫu có sẵn (00:30 – 01:45)

#### 1. Thử nghiệm trên lá có đốm bệnh nặng (`hu2.png`)
- **Hành động trên màn hình:**
  1. Giữ chọn mục "Ảnh mẫu có sẵn".
  2. Tại ô "Chọn ảnh mẫu", chọn: `Có đốm · hu2.png`.
  3. Bấm nút màu đỏ/xanh: **[Phân tích ảnh]**.
- **Lời thoại thuyết minh:**
  > *"Trước tiên, em xin chọn một ảnh mẫu lá bị bệnh nặng có tên là hu2.png và bấm Phân tích ảnh. Hệ thống chỉ mất chưa tới 1 giây để xử lý xong. Như Thầy/Cô có thể quan sát, trên giao diện hiển thị ngay 4 thẻ chỉ số quan trọng:*
  > *- Diện tích phiến lá: 558,256 pixel.*
  > *- Diện tích đốm bệnh: 171,535 pixel.*
  > *- Tỷ lệ tổn thương là 30.73%.*
  > *- Mức độ cảnh báo: Nặng (Severe Stage) do tỷ lệ vượt ngưỡng 20%.*
  > *Bên dưới là 4 hình ảnh trực quan thể hiện rõ luồng xử lý: Ảnh đã tiền xử lý tăng tương phản; Mask phiến lá tách bằng GrabCut đã lấp đầy các vết thủng; Mask đốm phát hiện bằng không gian màu HSV kết hợp LAB; và cuối cùng là ảnh gốc được khoanh đường viền đỏ quanh toàn bộ các vết hoại tử."*

#### 2. Thử nghiệm trên lá bình thường khỏe mạnh (`4.png`)
- **Hành động trên màn hình:**
  1. Chọn lại ô danh sách ảnh mẫu sang: `Bình thường · 4.png`.
  2. Lưu ý chỉ chuột vào màn hình: Kết quả cũ của ảnh `hu2` biến mất ngay lập tức (cơ chế chống nhầm lẫn dữ liệu).
  3. Bấm **[Phân tích ảnh]**.
- **Lời thoại thuyết minh:**
  > *"Tiếp theo, em chuyển sang thử nghiệm ảnh mẫu lá khỏe mạnh 4.png. Ngay khi em đổi ảnh, hệ thống tự động xóa kết quả cũ để tránh nhầm số liệu. Em bấm Phân tích ảnh: Kết quả hiển thị Diện tích đốm là 0 pixel, Tỷ lệ đốm là 0.00%, Cấp độ: Bình thường (Healthy). Mask đốm đen hoàn toàn và ảnh không bị bất kỳ đường viền đỏ nhận nhầm nào. Điều này chứng minh thuật toán kiểm soát hiện tượng nhận diện nhầm cực kỳ tốt."*

---

### ⏱️ Phần 3: Demo tính năng tải ảnh từ máy tính cá nhân (01:45 – 02:45)
- **Hành động trên màn hình:**
  1. Click chọn radio button: **"Tải ảnh của bạn"**.
  2. Bấm vào nút Browse files / kéo thả một file ảnh từ máy tính (ví dụ chọn ảnh `data/spotted/hu1.png` hoặc ảnh ngoài bất kỳ).
  3. Ảnh xem trước (preview) hiện lên.
  4. Bấm **[Phân tích ảnh]**.
- **Lời thoại thuyết minh:**
  > *"Ngoài các ảnh mẫu có sẵn, hệ thống hỗ trợ người dùng tải ảnh chụp bất kỳ từ máy tính cá nhân hoặc điện thoại. Em chuyển sang chế độ ‘Tải ảnh của bạn’, tải lên ảnh hu1.png và bấm Phân tích. Kết quả xác định được diện tích lá 377,274 pixel, diện tích đốm 16,188 pixel, tỷ lệ tổn thương 4.29% và xếp loại ở mức Nhẹ (Low / Early Stage). Các vết đốm màu nâu và vàng cháy ở mép lá được khoanh viền đỏ chính xác 100% bên trong phiến lá."*

---

### ⏱️ Phần 4: Demo tính năng xuất kết quả & Tải báo cáo CSV/PNG (02:45 – 03:15)
- **Hành động trên màn hình:**
  1. Cuộn chuột xuống phần **"Tải ảnh kết quả và số liệu"**.
  2. Click tải thử file: **"Tải ảnh khoanh đỏ"** $\rightarrow$ File PNG tải về máy.
  3. Click tải thử file: **"Tải số liệu CSV"** $\rightarrow$ Mở file CSV vừa tải lên cho thấy các cột `leaf_pixels`, `spot_pixels`, `spot_percent`, `grade`.
- **Lời thoại thuyết minh:**
  > *"Cuối cùng, hệ thống cung cấp tính năng xuất dữ liệu phục vụ báo cáo và lưu trữ hồ sơ vườn. Người dùng có thể tải về từng ảnh mask PNG độ phân giải chuẩn, ảnh khoanh đỏ và tệp số liệu CSV để phục vụ thống kê hoặc theo dõi diễn biến bệnh hại theo thời gian."*

---

### ⏱️ Phần 5: Kết thúc và cảm ơn (03:15 – 03:30)
- **Hành động trên màn hình:** Cuộn chuột lên đầu trang web, dừng hình ảnh tại giao diện tổng quan.
- **Lời thoại thuyết minh:**
  > *"Video demo chương trình của nhóm đến đây là kết thúc. Hệ thống đã đáp ứng đầy đủ yêu cầu kỹ thuật và phân công của đề tài học phần Xử lý ảnh. Chúng em xin chân thành cảm ơn Thầy/Cô đã theo dõi!"*
