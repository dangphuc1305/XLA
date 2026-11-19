# Kiểm tra trực tiếp giao diện — phần Nhựt

Hoàn tất trên Chrome bằng kỹ năng computer-use, sau khi các lần chuyển trang trước đó bị dừng. Ứng dụng chạy cục bộ tại `http://127.0.0.1:8501/`.

## Các thao tác đã xác nhận

| Thao tác | Kết quả quan sát |
| --- | --- |
| Mở trang kết quả có sẵn | Hiển thị hu5, 636.407 pixel lá, 12.752 pixel đốm, 2,00% |
| Đổi ảnh mẫu từ hu5 sang hu2 | Số liệu và ảnh kết quả cũ được xóa; yêu cầu bấm phân tích lại |
| Phân tích ảnh mẫu hu2 | Hiển thị 558.256 pixel lá, 171.535 pixel đốm, 30,73%, mức theo ngưỡng Nặng |
| Quan sát bốn ảnh hu2 | Ảnh tiền xử lý, mask lá, mask đốm và ảnh khoanh đỏ đều hiển thị |
| Tải CSV hu2 | Đã lưu; nội dung khớp số liệu trên giao diện |
| Chuyển sang nguồn ảnh tải lên, chưa chọn tệp | Kết quả cũ được xóa và nút phân tích bị vô hiệu hóa |
| Chọn tệp qua hộp thoại hệ thống | Đã tải `data/spotted/hu1.png`, hiển thị tên tệp và ảnh xem trước |
| Phân tích tệp đã tải lên | Hiển thị 377.274 pixel lá, 16.188 pixel đốm, 4,29%, mức theo ngưỡng Nhẹ |
| Tải các ảnh PNG hu1 | Cả mask lá, mask đốm và ảnh khoanh đỏ đã lưu thành công |
| Tải CSV hu1 | Đã lưu thành công và khớp các PNG tải từ cùng giao diện |

## Minh chứng đã lưu

Các tệp trong `results/nhut/ui_checks/` được tải bằng nút trên giao diện trình duyệt:

- `hu1_leaf_mask.png`
- `hu1_spot_mask.png`
- `hu1_analyzed.png`
- `hu1_metrics.csv`
- `verification.txt`: kết quả đọc lại và kiểm tra các tệp.

CSV hu2 tải trước đó nằm tại `C:/Users/PC/Downloads/spotted_hu2_metrics.csv`.

## Kiểm tra lại dữ liệu tải xuống

Đã giải mã cả ba PNG bằng OpenCV và đọc CSV UTF-8. Mask lá và mask đốm là ảnh một kênh `uint8`, cùng kích thước ảnh khoanh đỏ, chỉ chứa 0 và 255. Kết quả đếm pixel:

| Chỉ số | Giá trị |
| --- | ---: |
| Pixel lá | 377.274 |
| Pixel đốm | 16.188 |
| Tỷ lệ đã làm tròn | 4,29% |
| Pixel đốm ngoài mask lá | 0 |

Số liệu CSV trùng với giá trị đếm trên mask PNG. Không phát hiện lỗi chức năng mới trong các luồng vừa kiểm tra, vì vậy không thay đổi thuật toán hoặc giao diện ở lần tiếp tục này. Đã cập nhật trạng thái bàn giao để ghi nhận kiểm tra trực tiếp hoàn tất.

Đây là kiểm tra chức năng và hợp đồng dữ liệu. Giới hạn về nhận nhầm màu/gân/bóng và việc chưa có mask chuẩn để đo độ chính xác vẫn được ghi trong Chương 3.
