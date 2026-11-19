# Durian Leaf Spot Detection and Severity Analysis
## Hệ Thống Phát Hiện Đốm Lá Sầu Riêng và Chẩn Đoán Bệnh Bằng AI & Computer Vision

Hệ thống kết hợp **Thị giác máy tính (OpenCV)** và **Trí tuệ nhân tạo (Deep Learning MobileNetV3 / ONNX)** nhằm phát hiện đốm bất thường, định lượng tỷ lệ diện tích tổn thương và nhận diện phân loại chính xác các bệnh hại phổ biến trên lá sầu riêng tại Việt Nam.

---

## 👥 Phân Công Nhiệm Vụ Trong Nhóm
- **Lê Hữu Hào:** Chuẩn bị dữ liệu, Tiền xử lý (`src/preprocessing.py`), Phân đoạn tách phiến lá GrabCut (`src/leaf_segmentation.py`), Tích hợp CLI (`main.py`), Kiểm thử kỹ thuật và Biên soạn Chương 1–2.
- **Trần Minh Nhựt:** Thuật toán phát hiện đốm bất thường đa không gian màu HSV & LAB (`src/spot_detection.py`), Bàn giao kỹ thuật Chương 3.
- **Đặng Hoàng Phúc:** Thuật toán phân cấp mức độ nghiêm trọng (`src/severity_analysis.py`), Mô hình AI Deep Learning MobileNetV3 (`src/ai_model.py`), Cơ sở dữ liệu SQLite (`src/database.py`) và Giao diện ứng dụng Streamlit (`app.py`).

---

## 🧠 Thành Tựu Huấn Luyện Mô Hình AI Deep Learning
- **Kiến trúc mô hình:** MobileNetV3-Small (Transfer Learning tối ưu hóa cho thiết bị biên).
- **Bộ dữ liệu khoa học:** Mendeley Data DOI `10.17632/pxzvksbwnj.5` (2,595 ảnh thực tế tại Việt Nam, chuẩn hóa 6 lớp: `Leaf_Algal`, `Leaf_Blight`, `Leaf_Colletotrichum`, `Leaf_Healthy`, `Leaf_Phomopsis`, `Leaf_Rhizoctonia`).
- **Độ chính xác tổng thể (Test Accuracy trên 389 ảnh độc lập):** `88.43%`.
- **F1-Score (Macro Avg):** `88.49%` | **Weighted Avg:** `88.36%`.
- **F1-Score lớp Bệnh Đốm Lá Phomopsis (Đốm Mắt Cua):** `96.18%` (Recall: `100.0%`).
- **Thời gian suy luận (Inference Latency):** `18.21 ms / ảnh` (đáp ứng thời gian thực).
- **Định dạng triển khai:** ONNX Runtime (`models/mobilenetv3_durian.onnx` - 6.1 MB) và PyTorch (`models/mobilenetv3_durian.pth` - 6.2 MB).

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
durian-leaf-spots/
├── data/
│   ├── normal/               # 5 ảnh thật lá sầu riêng bình thường đồ án gốc
│   ├── spotted/              # 5 ảnh thật lá sầu riêng có đốm đồ án gốc
│   ├── mendeley_samples/     # 12 ảnh thực tế đại diện 6 lớp bệnh từ Mendeley
│   └── diagnosis_history.db  # Cơ sở dữ liệu SQLite lưu lịch sử chẩn đoán
├── dataset/
│   └── mendeley_durian/      # Bộ dữ liệu khoa học Mendeley 2,595 ảnh (train, val, test)
├── models/
│   ├── mobilenetv3_durian.onnx    # Mô hình ONNX siêu nhẹ (6.1 MB)
│   ├── mobilenetv3_durian.pth     # Trọng số mô hình PyTorch (6.2 MB)
│   └── evaluation_metrics.json    # Báo cáo đánh giá độc lập (Acc, F1, Latency, Confusion Matrix)
├── docs/
│   ├── BAO_CAO_TONG_HOP.md   # Toàn văn Báo cáo Tổng hợp hoàn chỉnh (Chương 1, 2, 3, 4)
│   ├── BAN_GIAO_HAO.md       # Biên bản bàn giao kỹ thuật của Hào cho Nhựt và Phúc
│   ├── BAN_GIAO_NHUT.md      # Biên bản bàn giao kỹ thuật của Nhựt cho Hào và Phúc
│   ├── BAN_GIAO_PHUC.md      # Biên bản bàn giao kỹ thuật của Phúc cho Hào và Nhựt
│   ├── CHUONG_1_2_HAO.md     # Nội dung báo cáo lý thuyết & phương pháp Chương 1 và 2 (Hào)
│   ├── CHUONG_3_NHUT.md      # Nội dung báo cáo thuật toán phát hiện đốm Chương 3 (Nhựt)
│   ├── CHUONG_4_PHUC.md      # Nội dung báo cáo thực nghiệm, kiểm thử & kết luận Chương 4 (Phúc)
│   ├── HUONG_DAN_QUAY_VIDEO_DEMO.md # Kịch bản chi tiết & lời thoại quay video demo nộp bài
│   └── HUONG_DAN_THUYET_TRINH_VA_SLIDE.md # Cấu trúc 12 slide bảo vệ đồ án & bộ câu hỏi phản biện
├── scripts/
│   ├── train_durian_model.py # Script huấn luyện và đánh giá mô hình MobileNetV3
│   ├── verify_dataset.py     # Script kiểm tra tính toàn vẹn bộ dữ liệu
│   └── evaluate_spots.py     # Script tái tạo minh chứng kiểm thử OpenCV
├── src/
│   ├── __init__.py           # Khởi tạo package
│   ├── preprocessing.py      # Tiền xử lý (đọc an toàn, resize giữ tỷ lệ, GaussianBlur 3x3, CLAHE)
│   ├── leaf_segmentation.py  # Phân đoạn tách lá (GrabCut + Otsu, lấp đầy lỗ đốm, Sanity Checks)
│   ├── spot_detection.py     # Phát hiện đốm bệnh trên phiến lá đa không gian màu (Nhựt)
│   ├── severity_analysis.py  # Tính tỷ lệ diện tích đốm/lá và xếp loại cấp độ bệnh (Phúc)
│   ├── disease_classifier.py # Chẩn đoán tác nhân bệnh thực vật theo hình thái học vi thể
│   ├── ai_model.py           # Module AI Deep Learning (Quality gatekeeper, Top-3 Softmax, phác đồ điều trị)
│   └── database.py           # Quản lý cơ sở dữ liệu SQLite lưu lịch sử chẩn đoán
├── app.py                    # Ứng dụng giao diện Web trực quan Dark Mode (Streamlit)
├── main.py                   # Script chạy kiểm tra phân tích bằng dòng lệnh (CLI)
├── .gitignore                # Cấu hình loại trừ file tạm và môi trường ảo
└── README.md                 # Tài liệu hướng dẫn sử dụng dự án
```

---

## ⚙️ Cài Đặt Trên Windows (Installation)

### 1. Yêu cầu môi trường:
- Hệ điều hành: Windows 10/11
- Python 3.10 trở lên (khuyến nghị tạo môi trường ảo `.venv`)

### 2. Kích hoạt môi trường và cài đặt thư viện:
```powershell
# Tạo môi trường ảo (nếu chưa có)
python -m venv .venv

# Kích hoạt môi trường ảo trên PowerShell
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 🚀 Cách Chạy Chương Trình (Usage)

### 1. Phân tích qua dòng lệnh (CLI):
```powershell
# Chạy với tham số vị trí
.\.venv\Scripts\python.exe main.py data\normal\1.png
.\.venv\Scripts\python.exe main.py data\spotted\hu1.png

# Chạy với cờ --image và thư mục lưu tùy chọn
.\.venv\Scripts\python.exe main.py --image data\spotted\hu2.png --output results

# Chạy với vùng ROI có kiểm soát (x y w h) cho ảnh nền phức tạp
.\.venv\Scripts\python.exe main.py data\spotted\hu1.png --roi 100 100 1000 1000
```

### 2. Chạy ứng dụng giao diện web (Streamlit):
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Sau khi khởi chạy, trình duyệt web sẽ mở tại địa chỉ `http://localhost:8501`. Người dùng có thể kéo thả ảnh lá sầu riêng để xem kết quả phân tích trực quan 4 bước.

Giao diện có hai nguồn ảnh: **Ảnh mẫu có sẵn** (10 ảnh trong dự án) và **Tải ảnh của bạn**. Chọn ảnh rồi bấm **Phân tích ảnh** để xem diện tích lá, diện tích đốm, tỷ lệ và ảnh khoanh đỏ. Mục **Tải ảnh kết quả và số liệu** cho phép tải hai mask PNG, ảnh khoanh đỏ và bảng CSV. Khi chọn ảnh khác, kết quả cũ được ẩn để tránh nhầm số liệu. Tỷ lệ rất nhỏ nhưng khác 0 được hiển thị `<0.01%`.

CLI kiểm tra lỗi ghi ảnh và hỗ trợ đường dẫn tiếng Việt bằng `src/image_io.py`; chỉ báo đã lưu sau khi cả ba ảnh được ghi thành công.

---

## 🔍 Cách Xem Kết Quả (Outputs)

Với mỗi ảnh đầu vào `<name>.<ext>`, hệ thống tự động xuất và lưu 3 ảnh vào thư mục `results/`:
1. `<name>_analyzed.<ext>`: Ảnh tiền xử lý được vẽ đường viền đỏ (`cv2.drawContours`) bao quanh các vết đốm phát hiện được.
2. `<name>_leaf_mask.png`: Mask nhị phân của phiến lá (`uint8`), trong đó vùng lá là màu trắng (`255`), hậu cảnh là màu đen (`0`).
3. `<name>_spot_mask.png`: Mask nhị phân các vết đốm bệnh (`uint8`), đảm bảo 100% nằm gọn bên trong vùng phiến lá hợp lệ.

Bảng kết quả in ra terminal bao gồm:
- **Leaf Area (px):** Tổng số pixel của phiến lá.
- **Spot Area (px):** Tổng số pixel của các vết đốm bệnh.
- **Severity Percentage:** Tỷ lệ phần trăm diện tích nhiễm bệnh: $\frac{\text{Spot Area}}{\text{Leaf Area}} \times 100\%$.
- **Severity Grade:** Xếp loại mức độ (Healthy, Low, Moderate, Severe).

---

## 📸 Điều Kiện Ảnh Đầu Vào & Giới Hạn Kỹ Thuật

### 1. Điều kiện ảnh được hỗ trợ tự động:
- Ảnh chụp có **một lá sầu riêng là đối tượng chính** nằm ở khu vực trung tâm khung hình.
- Nền chụp: Hỗ trợ nền sáng phòng thí nghiệm (giấy trắng, bàn trơn) hoặc nền vườn tự nhiên (đất, tán cây có độ sâu trường ảnh vừa phải).
- Ánh sáng: Đủ sáng, không bị cháy sáng toàn bộ hoặc tối mịt.

### 2. Các lỗi thường gặp và cách xử lý:
- **Lỗi `LeafSegmentationError: Vùng lá quá lớn (> 85%)`:** Xảy ra khi ảnh chụp góc quá cận sát mép hoặc thuật toán bị bắt nhầm nền xung quanh.  
  *Khắc phục:* Lùi khoảng cách chụp để phiến lá chiếm khoảng 30% - 60% khung hình hoặc dùng tham số `--roi x y w h`.
- **Lỗi `LeafSegmentationError: Mask lá chạm mép bất thường`:** Xảy ra khi tán cây nền ngoài dính chặt vào mép ảnh.  
  *Khắc phục:* Chụp ảnh với hậu cảnh thoáng hơn hoặc đặt lá lên bề mặt đơn sắc.
- **Lỗi nhầm lẫn màu mặt dưới lá:** Mặt dưới lá sầu riêng có màu vàng nâu đồng tự nhiên; cần lưu ý module phát hiện đốm có thể nhầm lẫn vùng này nếu không được điều chỉnh thích ứng.

### 3. Giới hạn hệ thống:
- Hệ thống hiện tại **chưa hỗ trợ tự động tách rời từng lá trong chùm nhiều lá đan xen chồng lấn**.
- Bộ dữ liệu hiện tại gồm **10 ảnh thật do Hào cung cấp**: 5 ảnh bình thường và 5 ảnh có đốm. Cả 10 ảnh đã chạy qua CLI. Hai ảnh `normal/1.png` và `normal/2.png` vừa được thay bằng ảnh một lá riêng; cả hai đã chạy lại và mask nhìn chung theo được viền lá. Chưa có mask chuẩn để tính độ chính xác định lượng.
