# Chương 3. Phương pháp và thuật toán phát hiện đốm bất thường

**Người thực hiện:** Trần Minh Nhựt. **Mã nguồn:** `src/spot_detection.py`.

## 3.1. Mục tiêu và phạm vi

Chức năng nhận ảnh lá đã tiền xử lý và mask phiến lá từ phần Hào, xác định các vùng có màu nâu, vàng hoặc đen bất thường, rồi tạo mask đốm để Phúc tính tỷ lệ diện tích. Kết quả hiển thị dùng đường viền đỏ giúp người dùng đối chiếu vị trí phát hiện với ảnh đầu vào.

Thuật toán sử dụng OpenCV và NumPy, không cần mô hình học máy. Những vùng được phát hiện là vùng bất thường về màu và hình dạng theo các quy tắc đặt ra. Thuật toán không phân biệt nguyên nhân sinh học gây đốm và không xác định tên bệnh.

## 3.2. Dữ liệu đầu vào và đầu ra

Hàm có chữ ký:

```python
def detect_spots(processed_bgr: np.ndarray,
                 leaf_mask: np.ndarray) -> np.ndarray:
    ...
```

`processed_bgr` là ảnh BGR của OpenCV, kiểu `uint8`, kích thước `(H, W, 3)`. `leaf_mask` là ảnh một kênh `uint8` `(H, W)`, trắng 255 là lá và đen 0 là nền. Cả hai đã cùng hệ tọa độ sau bước resize/tiền xử lý. Mask lá cần bao gồm cả phần tổn thương; phần lá không có trong mask đầu vào không thể được khôi phục bởi chức năng phát hiện đốm.

Đầu ra duy nhất `spot_mask` là ảnh một kênh `uint8` `(H, W)`. Trắng 255 là đốm và đen 0 là vùng khác. Mask phải thỏa:

\[
\forall (x,y),\quad M_{lá}(x,y)=0 \Rightarrow M_{đốm}(x,y)=0.
\]

Hàm kiểm tra kiểu dữ liệu, số kênh, kích thước, ảnh rỗng và giá trị nhị phân. Mask lá hợp lệ nhưng không có pixel lá sẽ trả mask đen, không phát sinh phép chia cho không. Hàm không tự chuyển mask 0/1 thành 0/255 nhằm phát hiện sớm lỗi tích hợp.

## 3.3. Quy trình tổng quát

1. Kiểm tra đầu vào và xác định vùng lá.
2. Chuyển ảnh BGR sang HSV và LAB.
3. Ước lượng màu tham chiếu của chính phiến lá và nền màu cục bộ.
4. Ngưỡng màu nâu, vàng và đen; kết hợp với độ lệch màu/độ sáng.
5. AND với mask lá, sau đó Opening và Closing.
6. Lọc thành phần nhỏ, hình dài mảnh giống gân và nhiễu nhỏ sát biên.
7. AND lần cuối, trả mask đốm; bên gọi lấy contour để khoanh đỏ.

## 3.4. Chuyển đổi không gian màu

### 3.4.1. HSV

HSV tách sắc độ H, độ bão hòa S và độ sáng V nên thuận tiện mô tả vùng màu. Trong ảnh `uint8` của OpenCV, H nằm trong `[0,179]`, S và V nằm trong `[0,255]`. Các giới hạn H trong mã dùng thang OpenCV, không phải số độ `[0,360]`.

Các mask sơ bộ được tạo bằng `cv2.inRange`:

| Nhóm | Ngưỡng HSV thấp | Ngưỡng HSV cao | Vai trò |
| --- | --- | --- | --- |
| Nâu, nâu đỏ | `(0, 50, 25)` | `(25, 255, 255)` | Tìm vùng nâu có bão hòa đủ lớn |
| Vàng | `(20, 65, 50)` | `(38, 255, 255)` | Giữ cả vùng vàng sáng |
| Tối/đen | `(0, 0, 0)` | `(179, 255, 55)` | Tìm vùng tối kể cả khi bão hòa thấp |

Chỉ ngưỡng HSV chưa đủ: mặt dưới lá sầu riêng có thể vàng đồng tự nhiên, gân cũng có thể vàng, còn bóng đổ làm lá xanh bị tối. Vì vậy cần thêm thông tin LAB và màu tham chiếu.

### 3.4.2. LAB

LAB gồm độ sáng L và hai trục màu a, b. Trong biểu diễn OpenCV `uint8`, a và b có mức trung tính xấp xỉ 128; a thấp thiên xanh, a cao thiên đỏ, b cao thiên vàng. Các giá trị trong mã là giá trị đã mã hóa trên thang 8 bit, không dùng trực tiếp thang CIE có giá trị âm.

Các kênh được chuyển sang `float32` trước khi tính hiệu. Nếu trừ trên `uint8`, kết quả âm có thể quay vòng thành giá trị dương lớn, gây sai việc đánh giá độ tương phản.

## 3.5. Thích ứng theo màu nền của lá

### 3.5.1. Chọn màu tham chiếu

Tập mô xanh tham chiếu phải đồng thời nằm trong mask lá và thỏa H từ 35 đến 95, S ít nhất 40, V ít nhất 45, a nhỏ hơn 128. Nếu số pixel này đạt ít nhất 10% diện tích lá, dùng trung vị a và L của tập xanh. Cách này cho phép giữ một mảng hoại tử tương đối lớn khi vẫn còn đủ mô xanh làm chuẩn.

Trên lá chủ yếu vàng đồng, dùng trung vị a và L của toàn bộ vùng lá. Một mặt lá vàng đồng đồng đều vì thế không bị coi là tổn thương chỉ vì màu nằm trong dải nâu/vàng.

### 3.5.2. Nền cục bộ có trọng số mask

Với một kênh màu C, nền cục bộ được tính:

\[
\bar C(x,y)=\frac{G_\sigma*(C\cdot W)}{\max(G_\sigma*W,10^{-6})},
\quad W=\mathbf{1}[M_{lá}=255].
\]

Ở đây `*` biểu diễn phép tích chập Gaussian. `sigma = max(2, sqrt(A_lá) × 0,025)`; `A_lá` là số pixel lá. Các pixel nền được gán trọng số 0 nên không làm sai nền tham chiếu ở rìa, kể cả khi ảnh đầu vào đã được tô đen nền.

Hai đại lượng sử dụng là `local_L - L` và `a - local_a`. Đại lượng thứ nhất dương lớn khi một điểm tối hơn lân cận; đại lượng thứ hai dương lớn khi một điểm đỏ hơn lân cận.

### 3.5.3. Kết hợp điều kiện

Nếu có đủ mô xanh, ứng viên nâu cần a ít nhất `max(122, a_tham_chiếu + 12)`. Ứng viên vàng cần a ít nhất `max(105, a_tham_chiếu + 12)`, b ít nhất 145 và tương phản cục bộ ít nhất 8 ở sắc đỏ hoặc độ tối. Điều kiện tương phản giúp loại vùng phản quang/giọt nước chỉ có màu vàng sáng.

Nếu không đủ mô xanh, ứng viên nâu/vàng cần a cao hơn trung vị lá ít nhất 8. Nhánh linh hoạt hơn cho phép độ lệch a ít nhất 4 khi điểm đó đồng thời tối hơn lân cận ít nhất 15 và đỏ hơn lân cận ít nhất 4. Mục đích là giữ các đốm cục bộ trên mặt dưới lá vàng đồng.

Ứng viên tối/đen được nhận khi màu không còn thuộc nhóm xanh thông thường và L thấp hơn L tham chiếu ít nhất 30. Một điểm gần đen có V không quá 25 cũng được nhận nếu tối hơn lân cận ít nhất 12. Với lõi hoại tử xám nằm sát vùng nâu/vàng, mã bổ sung ứng viên có S không quá 65, V không quá 110, L thấp hơn màu tham chiếu ít nhất 35 và tối hơn lân cận ít nhất 25. Ràng buộc khoảng cách chỉ mở rộng quanh ứng viên nâu/vàng, giúp nhận lõi tối của đốm như ở `hu4` mà hạn chế bắt bóng tối độc lập trên lá. Để giảm nhận nhầm bóng đổ sát silhouette, pixel tối nằm trong dải biên `max(3, round(sqrt(A_lá) × 0,055))` chỉ được giữ nếu đồng thời thuộc ứng viên màu nâu/vàng bất thường. Quy tắc này ưu tiên loại viền bóng tối, nhưng một phần hoại tử đen sát mép mà không có màu nâu/vàng có thể bị bỏ sót. Bóng nằm sâu bên trong lá vẫn có thể bị nhận nhầm nếu màu của nó gần giống màu đốm.

Trước khi tạo ứng viên, mở mask lá bằng ellipse có cạnh xấp xỉ `sqrt(A_lá) × 0,04` để giới hạn phép dò vào phần phiến đủ rộng. Tiếp đó PCA so bề rộng hai đầu theo trục dài; nếu một đầu hẹp bất đối xứng rõ rệt, phần cuống ở đầu đó bị loại khỏi ROI phát hiện. Mask lá đầu vào không bị sửa, và mask đốm vẫn được AND với mask lá gốc. Heuristic này có thể bỏ mất tổn thương nằm đúng ở cuống hoặc chóp lá.

## 3.6. Lọc nhiễu và hậu xử lý

### 3.6.1. Giới hạn bằng mask lá

Các mask màu được hợp bằng OR, chuyển thành `uint8` 0/255 rồi áp dụng:

```python
candidates = cv2.bitwise_and(candidates, leaf_mask)
```

Phép AND được lặp lại sau từng bước hình thái và ngay trước khi trả về. Đây là ràng buộc dữ liệu bắt buộc, vì phép dilation trong hậu xử lý có thể tạo pixel trắng ra ngoài mask lá.

### 3.6.2. Opening và Closing

Sử dụng phần tử cấu trúc ellipse 3×3:

- **Opening** gồm erosion rồi dilation, loại chấm nhiễu đơn lẻ và các đường quá mảnh.
- **Closing** gồm dilation rồi erosion, lấp các lỗ nhỏ và nối khe đứt hẹp trong một vùng đốm.

Kích thước kernel nhỏ giúp giữ các đốm rải rác. Hàm không tô kín toàn bộ contour vì thao tác đó có thể biến cả vùng lá lành trong một vòng đốm thành diện tích tổn thương.

### 3.6.3. Lọc thành phần liên thông

Sử dụng liên thông 8 với `cv2.connectedComponentsWithStats`. Một thành phần phải có ít nhất `max(48, round(A_lá × 0,00008))` pixel để được giữ. Ngưỡng này bỏ các chấm rất nhỏ thường do gân/giọt nước, đồng thời thay đổi theo kích thước ảnh.

Để hạn chế gân, lấy hình chữ nhật xoay nhỏ nhất bằng `cv2.minAreaRect`. Nếu tỷ số cạnh dài/cạnh ngắn lớn hơn 4, kiểm tra thêm độ mảnh. Thành phần bị loại khi cạnh ngắn hoặc phân vị 75% bán kính bên trong không vượt `max(4, sqrt(A_lá) × 0,008)`. Bán kính bên trong lấy bằng distance transform, giúp kiểm tra những đường gân hơi cong mà hình chữ nhật bao có bề rộng lớn hơn độ dày thực.

Đối với nhiễu cuống/viền, tính khoảng cách đến biên mask lá. Loại thành phần nhỏ hơn 0,35% diện tích lá nếu ít nhất 70% pixel nằm trong dải biên `max(3, sqrt(A_lá) × 0,025)`; với thành phần nhỏ hơn 3%, yêu cầu 85% pixel nằm trong dải này. Với nhánh mảnh dính vào một đốm lớn, chỉ hỗ trợ pixel sát biên trong phạm vi `max(4, band // 3) + ceil(bán_kính_lõi)` để tránh lõi bệnh khôi phục một đường bóng chạy xa dọc silhouette. Không xói toàn bộ mask lá nên phần rộng của vùng cháy chạm mép vẫn có thể được giữ.

Đây là các quy tắc hình học kinh nghiệm. Một đốm thật dài mảnh hoặc nhỏ sát mép có thể bị loại nhầm; cần cân đối ngưỡng bằng dữ liệu có gán nhãn.

Sau bước lọc thành phần, chương trình kiểm tra các lỗ kín bên trong từng vùng đốm. Chỉ lấp lỗ nhỏ hơn một nửa thành phần khi trung vị L của lỗ thấp hơn vòng bao ít nhất 12 mức; lỗ sáng do phản quang hoặc mô lá được giữ. Quy tắc này khôi phục lõi hoại tử tối bị bỏ bởi ngưỡng màu mà không tô kín mọi vùng lõm của contour.

## 3.7. Hiển thị và bàn giao kết quả

`main.py` và `app.py` nhận duy nhất mask trả về, sau đó tự lấy contour từ chính mask đó. Đường viền vẽ bằng màu BGR `(0, 0, 255)`, độ dày 2 pixel. Ảnh nền là ảnh đã tiền xử lý để cùng kích thước và tọa độ với mask.

Phần Phúc nhận `calculate_severity(leaf_mask, spot_mask)` và tính:

\[
P=\frac{\operatorname{countNonZero}(M_{đốm})}
        {\operatorname{countNonZero}(M_{lá})}\times100\%.
\]

Nếu mask lá rỗng, phần tính tỷ lệ trả 0 theo cách xử lý đang có. Thuật toán Nhựt không đặt ngưỡng phân cấp mức độ bệnh; chức năng này thuộc phần Phúc.

## 3.8. Kiểm thử và kết quả thực nghiệm

### 3.8.1. Kiểm thử tự động

Các kiểm thử tổng hợp xác minh màu nâu/vàng/đen tại vị trí biết trước, loại nền nâu, loại nhiễu đơn lẻ và gân mảnh, lấp lỗ nhỏ, giữ vùng cháy lớn và vùng dài đủ dày, xử lý lá đồng màu vàng đồng, bóng xanh tối, đốm sát mép và lỗ trong mask lá. Ngoài ra có kiểm tra đầu vào sai, mask rỗng, ảnh rất nhỏ, mảng không liên tục, bảo toàn đầu vào và tích hợp công thức tỷ lệ của Phúc.

Lệnh chạy: `python -m unittest discover -s tests -v`. Các ảnh tổng hợp phục vụ kiểm tra hành vi chương trình, không đại diện đầy đủ cho điều kiện tự nhiên.

### 3.8.2. So sánh trên 10 ảnh của nhóm

Hai thuật toán dùng cùng ảnh tiền xử lý và cùng mask lá cho mỗi ảnh. Bản cũ là ngưỡng HSV `[10,50,50]`–`[30,255,200]`, AND và Opening 3×3. Bản mới là hàm trong chương này. Mã đánh giá cố định seed GrabCut về 0; các kết quả có thể khác nhẹ so với những lần tách lá trước đó.

| Ảnh | Tỷ lệ đốm bản cũ (%) | Tỷ lệ đốm bản mới (%) | Pixel đốm ngoài lá mới |
| --- | ---: | ---: | ---: |
| normal/1.png | 0,8551 | 0,0000 | 0 |
| normal/2.png | 0,0000 | 0,0000 | 0 |
| normal/3.png | 79,9452 | 0,0000 | 0 |
| normal/4.png | 0,0000 | 0,0000 | 0 |
| normal/5.png | 0,0273 | 0,0000 | 0 |
| spotted/hu1.png | 76,8373 | 4,0981 | 0 |
| spotted/hu2.png | 26,6175 | 22,2333 | 0 |
| spotted/hu3.png | 3,9609 | 1,7572 | 0 |
| spotted/hu4.png | 5,7316 | 7,5662 | 0 |
| spotted/hu5.png | 1,2673 | 1,5716 | 0 |

Toàn bộ 10 ảnh được chạy lại sau khi bổ sung lọc bóng sát biên, ROI phiến lá để bỏ cuống hẹp, giảm nhận nhầm gân giữa và lọc thành phần nhỏ bám mép. Mọi mask đều có 0 pixel ngoài mask lá. Số liệu và ảnh đối chiếu hiện tại nằm trong `results/nhut/full_review/`; đây là tỷ lệ pixel được đánh dấu, không phải phần trăm dự đoán đúng.

Nguồn số liệu: `results/nhut/full_review/comparison.csv`.

Quan sát toàn bộ contact sheet cho thấy `normal/3.png` không còn pixel bị đánh dấu, các đốm ở `hu1`, `hu3`, `hu4`, `hu5` phần lớn được khoanh tại vùng nâu/đen quan sát được, và `hu2` không còn khoanh cuống hay dải bóng sát mép. Ở `hu4`, lõi hoại tử tối trong mảng bệnh lớn được bổ sung vào mask; bộ lọc tương phản cục bộ tiếp tục giảm vùng vàng phản quang/giọt nước bị đánh dấu. Chưa có mask chuẩn để xác nhận các vùng tăng/giảm có đúng hay không.

Một vài pixel lẻ còn xuất hiện khi thay đổi độ sáng; đốm sát mép hoặc nằm đúng gân có thể bị mất do các bộ lọc hình học. Các kết quả cần được trình bày cùng ảnh mask, tránh chỉ dựa vào tỷ lệ tổng.

### 3.8.3. Kiểm tra thay đổi độ sáng

Nhân ảnh đã tiền xử lý với các hệ số 0,75; 1; 1,25 rồi giới hạn giá trị về `[0,255]`. Giữ mask lá cố định để đánh giá riêng detector. Tổng cộng 30 lượt xử lý đều không có đốm ngoài lá. Dải tỷ lệ qua ba mức sáng trên một số ảnh:

| Ảnh | Tỷ lệ nhỏ nhất–lớn nhất (%) |
| --- | ---: |
| normal/1.png | 0,0000–0,0000 |
| normal/2.png | 0,0000–0,0000 |
| normal/3.png | 0,0000–0,0000 |
| normal/4.png | 0,0000–0,0000 |
| normal/5.png | 0,0000–0,0000 |
| spotted/hu1.png | 3,6053–4,4493 |
| spotted/hu2.png | 20,5424–22,2333 |
| spotted/hu3.png | 1,7000–1,8469 |
| spotted/hu4.png | 6,9800–7,8782 |
| spotted/hu5.png | 1,4664–1,7464 |

Nguồn: `results/nhut/full_review/lighting_checks.csv`. Thuật toán vẫn nhạy với độ sáng; phép thử mô phỏng này không chứng minh tính bất biến ánh sáng và không thay thế ảnh chụp thực tế dưới nhiều nguồn sáng.

## 3.9. Hạn chế và hướng cải thiện

Bộ dữ liệu hiện có 10 ảnh, chưa có mask đốm gán nhãn thủ công. Do đó chưa báo cáo được IoU, Dice, precision hay recall. Các ngưỡng được xây dựng và kiểm tra trên chính bộ ảnh nhỏ này, chưa có tập kiểm chứng độc lập nên không thể suy rộng kết quả.

Màu đồng đều bất thường trên cả lá có thể bị xem như màu tham chiếu và bỏ sót. Mô lá lành vàng tự nhiên, cuống và bóng tối có thể gần màu đốm; mô bệnh màu xanh cũng khó tách bằng các ngưỡng đang dùng. Kích thước kernel cố định và quy tắc lọc hình dạng có thể làm mất tổn thương rất nhỏ hoặc dài mảnh. Kết quả còn phụ thuộc trực tiếp vào chất lượng mask lá.

Bước tiếp theo là thu thập ảnh đa dạng, gán mask chuẩn, chia tập phát triển/kiểm chứng, đo IoU/Dice và khảo sát ngưỡng theo từng điều kiện chụp. Có thể đánh giá thêm đặc trưng kết cấu hoặc mô hình phân đoạn nếu dữ liệu đủ lớn.


