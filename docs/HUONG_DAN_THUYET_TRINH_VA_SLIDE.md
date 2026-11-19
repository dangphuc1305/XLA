# HƯỚNG DẪN XÂY DỰNG SLIDE THUYẾT TRÌNH VÀ BẢO VỆ ĐỒ ÁN TRƯỚC HỘI ĐỒNG

**Học phần:** Xử lý ảnh / Đồ án Công nghệ Thông tin  
**Đề tài:** Hệ thống chẩn đoán sâu bệnh trên lá sầu riêng bằng Thị giác máy tính và Trợ lý AI Nông nghiệp  
**Bộ tài liệu:** Cấu trúc 12 slide chuẩn điểm A+ & Bộ câu hỏi phản biện của Hội đồng.  

---

## PHẦN 1: CẤU TRÚC 12 SLIDE THUYẾT TRÌNH CHUẨN ĐỒ ÁN XUẤT SẮC

### Slide 1: Trang bìa (Title Slide)
- **Tiêu đề lớn:** HỆ THỐNG CHẨN ĐOÁN SÂU BỆNH TRÊN LÁ SẦU RIÊNG BẰNG THỊ GIÁC MÁY TÍNH VÀ TRỢ LÝ AI
- **Phụ đề:** Định lượng diện tích tổn thương, phân loại tác nhân nấm bệnh và tư vấn phác đồ điều trị thực nghiệm
- **Giảng viên hướng dẫn:** [Tên Giảng viên]
- **Nhóm sinh viên thực hiện:**
  - Lê Hữu Hào (Nhóm trưởng - Tiền xử lý & Phân đoạn GrabCut)
  - Trần Minh Nhựt (Thuật toán phát hiện đốm HSV & LAB)
  - Đặng Hoàng Phúc (Phân tích mức độ, Giao diện Dark Mode & Chatbot AI)

### Slide 2: Đặt vấn đề & Ý nghĩa thực tiễn (Motivation)
- Sầu riêng là cây ăn trái "tỷ đô" của nông nghiệp Việt Nam, nhưng cực kỳ mẫn cảm với dịch hại nấm lá (*Colletotrichum*, *Rhizoctonia*, *Phomopsis*).
- **Thực trạng:** Nhà vườn chủ yếu chẩn đoán bằng mắt thường cảm tính; khó ước lượng chính xác tỷ lệ tổn thương để định lượng liều lượng thuốc BVTV.
- **Giải pháp đề tài:** Tự động hóa quy trình chẩn đoán từ ảnh chụp với độ chính xác cao, tính toán tỷ lệ tổn thương theo pixel và tư vấn hoạt chất đặc trị.

### Slide 3: Kiến trúc hệ thống tổng thể (System Architecture)
- Vẽ sơ đồ khối 2 tầng:
  1. **Tầng Thị giác máy tính (Computer Vision Engine):** Ảnh đầu vào $\rightarrow$ Tiền xử lý (CLAHE + Gaussian) $\rightarrow$ Tách phiến lá (GrabCut) $\rightarrow$ Phát hiện đốm (HSV/LAB + Morphology) $\rightarrow$ Tính tỷ lệ diện tích $S = (A_{\text{đốm}} / A_{\text{lá}}) \times 100\%$.
  2. **Tầng Ứng dụng & Trí tuệ Nông nghiệp (Smart Agriculture Layer):**
     - Bộ phân loại tác nhân bệnh (Thán thư, Cháy lá, Đốm rong, Mắt cua).
     - Trợ lý AI nông nghiệp hỏi đáp theo ngữ cảnh.
     - Mô hình cảnh báo nguy cơ nấm bệnh thời tiết (FDRI).
     - Hệ thống theo dõi chuỗi thời gian tiến trình hồi phục của cây.

### Slide 4: Tiền xử lý & Tách phiến lá (Hào trình bày)
- **Kỹ thuật tiền xử lý:** Đọc ảnh Unicode đường dẫn tiếng Việt an toàn; resize giữ tỷ lệ aspect ratio; Gaussian blur 3x3 làm mịn; CLAHE cân bằng sáng cục bộ thích nghi.
- **Phân đoạn GrabCut:** 5 vòng lặp cắt đồ thị (Graph Cut) phân tách phiến lá khỏi đất vườn/nền nhân tạo.
- **Thuật toán lấp đầy lỗ thủng (Hole Filling):** Bảo toàn vùng hoại tử tối màu trong phiến lá không bị khoét rỗng.
- **Sanity Checks:** Tự động phát hiện lỗi khi diện tích lá bất thường ($<5\%$ hoặc $>85\%$) hoặc chạm viền quá mức.

### Slide 5: Thuật toán phát hiện đốm bất thường (Nhựt trình bày)
- **Đa không gian màu:** Phối hợp dải màu nâu hoại tử, vàng úa trên không gian HSV và độ chênh lệch sắc tố đỏ/vàng trên không gian LAB.
- **Lọc hình thái học:** Morphological Opening khử nhiễu hạt bụi li ti; Closing liền mạch các ổ bệnh.
- **Khử nhiễu hình học chuyên sâu:**
  - Lọc gân lá bằng tỷ lệ độ dài/rộng trục chính (Aspect Ratio).
  - Khử nhánh bóng mép lá bằng khoảng cách Euclidean Distance Transform.
- **Ràng buộc toán học:** $M_{\text{đốm}} = M_{\text{đốm}} \land M_{\text{lá}}$, cam kết 100% đốm nằm gọn bên trong lá.

### Slide 6: Phân loại bệnh cụ thể & Phác đồ điều trị (Phúc trình bày)
- Chuyển từ "đốm vô danh" sang chẩn đoán 4 bệnh chính:
  1. **Thán thư (*Colletotrichum gloeosporioides*):** Đốm tròn đồng tâm viền vàng $\rightarrow$ Hoạt chất: *Azoxystrobin, Difenoconazole*.
  2. **Cháy lá chết ngọn (*Rhizoctonia solani*):** Vết cháy loang mép lá lớn $\rightarrow$ Hoạt chất: *Validamycin, Hexaconazole*.
  3. **Đốm rong (*Cephaleuros virescens*):** Đốm nhung vàng cam gỉ sắt $\rightarrow$ Hoạt chất: *Gốc Đồng (Copper Oxychloride)*.
  4. **Đốm mắt cua (*Phomopsis durionis*):** Nhiều chấm nhỏ li ti có tâm sáng $\rightarrow$ Hoạt chất: *Propiconazole*.

### Slide 7: Trợ lý AI Nông nghiệp & Cảnh báo thời tiết FDRI
- **Context-Aware Chatbot:** Tự động đọc dữ liệu chiếc lá vừa quét để tư vấn trực tiếp cho nông dân.
- **Chỉ số rủi ro nấm bệnh (Fungal Disease Risk Index - FDRI):**
  - Kết hợp độ ẩm ($RH > 80\%$) và nhiệt độ ($25-30^\circ\text{C}$) theo mô hình dịch tễ học.
  - Đưa ra cảnh báo đỏ và hướng dẫn phun phòng trước các đợt mưa dầm.

### Slide 8: Quản lý lịch sử & Biểu đồ theo dõi tiến trình hồi phục
- Lưu trữ lịch sử khám bệnh theo từng mã cây/lô vườn.
- **Biểu đồ đường (Time-series Recovery Chart):** Minh chứng thực nghiệm cho thấy tỷ lệ đốm giảm dần từ $30.73\%$ $\rightarrow$ $11.50\%$ $\rightarrow$ $4.29\%$ $\rightarrow$ $0.00\%$ sau các đợt xử lý thuốc.

### Slide 9: Giao diện Web Dark Mode công thái học
- Trình diễn giao diện Dark Mode chuẩn công thái học:
  - Khắc phục triệt để lỗi mờ tương phản ở 4 thẻ chỉ số bằng bảng màu Slate/Emerald tương phản cao.
  - Bố cục khu vực tải ảnh và nút bấm cân đối, chuyên nghiệp.
  - 5 Tab chức năng trực quan, thân thiện với nông dân.

### Slide 10: Kết quả thực nghiệm trên 10 ảnh mẫu chụp thật
- Bảng tổng hợp số liệu 10 ảnh:
  - 5 ảnh lá có đốm: Phát hiện chính xác 100%, tỷ lệ tổn thương từ 2.0% đến 30.73%.
  - 5 ảnh lá bình thường: Kiểm soát false positive xuất sắc (4 ảnh đạt 0.00% đốm, 1 ảnh chỉ 0.54% vết mép lá).
  - Cải tiến ngoạn mục trên ảnh `3.png` (từ lỗi nhận nhầm 79.95% của thuật toán cũ xuống còn 0.12%).
- Kiểm tra độ bền vững trước 30 lượt thay đổi độ sáng ($\pm 25\%$).

### Slide 11: Video Demo / Live Demo sản phẩm
- Chạy trực tiếp hoặc phát video demo 3 phút: Chọn ảnh mẫu $\rightarrow$ Phân tích trong 0.8 giây $\rightarrow$ Xem kết quả 4 bước $\rightarrow$ Chat với Trợ lý AI $\rightarrow$ Xem biểu đồ hồi phục $\rightarrow$ Tải báo cáo CSV.

### Slide 12: Kết luận & Hướng phát triển tương lai
- **Kết luận:** Đề tài hoàn thành xuất sắc toàn bộ mục tiêu môn học, kết hợp hài hòa giữa lý thuyết Xử lý ảnh và giải pháp ứng dụng thực tiễn.
- **Hướng phát triển:** Mở rộng tập dữ liệu lớn (>500 ảnh) có Ground-Truth Mask chuẩn pixel; thử nghiệm mô hình Deep Learning (YOLOv8-Seg); đóng gói ứng dụng di động PWA/Android.

---

## PHẦN 2: BỘ CÂU HỎI PHẢN BIỆN CỦA HỘI ĐỒNG & CÂU TRẢ LỜI MẪU

#### ❓ Câu hỏi 1: *"Tại sao nhóm lựa chọn kỹ thuật Xử lý ảnh truyền thống (GrabCut, HSV, Morphology) mà không dùng ngay Mạng nơ-ron sâu (YOLO, Mask R-CNN)?"*
* **Trả lời mẫu:**
  > *"Thưa Thầy/Cô, nhóm chọn kỹ thuật Xử lý ảnh truyền thống vì 3 lý do cốt lõi:*
  > *1. **Tính chất học phần:** Đồ án nhằm áp dụng sâu các kiến thức nền tảng của môn học Xử lý ảnh (Không gian màu, Biến đổi hình thái, Phân đoạn đồ thị GrabCut, Bộ lọc không gian).*
  > *2. **Khả năng giải thích (Explainable AI):** Phương pháp truyền thống minh bạch về mặt toán học, từng bước lọc đốm hay khử gân lá đều có cơ sở định lượng rõ ràng, không phải là hộp đen (black box).*
  > *3. **Hiệu năng & Tài nguyên:** Thuật toán chạy mượt mà trên CPU thông thường chỉ mất dưới 1 giây mà không cần trang bị GPU đắt tiền, rất phù hợp để triển khai trên các thiết bị nhúng hoặc máy chủ nông nghiệp giá rẻ tại nông trại."*

#### ❓ Câu hỏi 2: *"Làm sao hệ thống phân biệt được vết cháy lá do nấm với vết cháy bìa lá do sinh lý thiếu Kali?"*
* **Trả lời mẫu:**
  > *"Thưa Thầy/Cô, nhóm dựa trên 2 đặc trưng thị giác quan trọng:*
  > *1. **Quầng viền chuyển tiếp (Halo border):** Vết cháy do nấm bệnh luôn có một đường viền quầng màu vàng tươi rõ nét ngăn cách giữa mô hoại tử nâu và mô lá khỏe do độc tố nấm tiết ra. Trong khi đó, cháy mép do thiếu Kali cháy đều dần từ mép vào và không có quầng vàng sắc nét.*
  > *2. **Độ đối xứng:** Thiếu Kali mang tính đối xứng đồng loạt trên phiến lá và xuất hiện nhiều ở tầng lá già, trong khi nấm bệnh phân bố bất định theo giọt bắn của nước mưa. Nhóm đã tích hợp kiến thức này vào cả thuật toán phân loại và Trợ lý AI nông nghiệp."*

#### ❓ Câu hỏi 3: *"Làm thế nào để hệ thống không nhận nhầm bóng đổ ở mép lá thành vết đốm bệnh?"*
* **Trả lời mẫu:**
  > *"Thưa Thầy/Cô, nhóm đã giải quyết vấn đề bóng mép lá bằng thuật toán biến đổi khoảng cách (Distance Transform). Vùng sát mép lá trong phạm vi 5-8 pixel được áp dụng cơ chế kiểm tra kép: Nếu một vùng tối sát mép không có một 'lõi đốm dày' phía bên trong hỗ trợ và không thỏa mãn điều kiện sắc tố nâu hoại tử trên kênh LAB ($a^*, b^*$), thuật toán sẽ tự động cắt tỉa (prune) nhánh bóng này. Nhờ đó, trên ảnh lá bình thường `2.png` và `4.png`, số pixel đốm nhận nhầm bằng đúng 0."*
