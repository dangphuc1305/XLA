"""Phát hiện vùng đốm bất thường trên lá sầu riêng — phần Trần Minh Nhựt.

Quy ước: ảnh màu BGR uint8; mask một kênh uint8, chỉ chứa 0 và 255.
Đây là thuật toán ngưỡng màu và hình thái học, không chẩn đoán tác nhân bệnh.
"""

import cv2
import numpy as np


def detect_spots(processed_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Nhận ảnh đã tiền xử lý và mask lá, trả về duy nhất ``spot_mask``.

    Parameters
    ----------
    processed_bgr : np.ndarray
        Ảnh màu OpenCV **BGR**, dtype uint8, kích thước (H, W, 3).
        Không cần tô đen nền trước khi gọi hàm; không đổi kích thước ở đây.
    leaf_mask : np.ndarray
        Mask nhị phân uint8 (H, W), cùng chiều cao/rộng với processed_bgr.
        Trắng (255) là lá, đen (0) là nền. Các vùng đốm phải thuộc mask lá.

    Returns
    -------
    np.ndarray
        spot_mask uint8 (H, W): trắng (255) là đốm, đen (0) là vùng khác.
        Luôn thỏa spot_mask[leaf_mask == 0] == 0. Mask lá rỗng trả mask đen.

    Raises
    ------
    TypeError
        Đầu vào không phải mảng NumPy uint8.
    ValueError
        Sai số kênh, ảnh rỗng, kích thước không khớp hoặc mask không nhị phân.

    Notes
    -----
    Ngưỡng dưới đây là giá trị khởi đầu cho dữ liệu của nhóm, cần hiệu chỉnh
    trên tập kiểm chứng có mask chuẩn khi thay camera/điều kiện chiếu sáng.
    Hàm không sửa hai mảng đầu vào và không trả contours. Có thể lấy contours
    từ kết quả bằng cv2.findContours khi cần khoanh đỏ để hiển thị.
    """
    # 1. Kiểm tra hợp đồng dữ liệu, tránh xử lý nhầm RGB, mask ba kênh hoặc
    # mask 0/1. Không thể tự phân biệt BGR và RGB từ các giá trị pixel:
    # bên gọi phải chuyển RGB -> BGR nếu ảnh đến từ PIL/Streamlit.
    if not isinstance(processed_bgr, np.ndarray) or not isinstance(leaf_mask, np.ndarray):
        raise TypeError("processed_bgr và leaf_mask phải là mảng NumPy.")
    if processed_bgr.dtype != np.uint8 or leaf_mask.dtype != np.uint8:
        raise TypeError("processed_bgr và leaf_mask phải có dtype uint8.")
    if processed_bgr.ndim != 3 or processed_bgr.shape[2] != 3 or processed_bgr.size == 0:
        raise ValueError("processed_bgr phải là ảnh BGR không rỗng, dạng (H, W, 3).")
    if leaf_mask.ndim != 2 or leaf_mask.shape != processed_bgr.shape[:2]:
        raise ValueError("leaf_mask phải có một kênh (H, W), cùng kích thước ảnh BGR.")
    if np.any((leaf_mask != 0) & (leaf_mask != 255)):
        raise ValueError("leaf_mask chỉ được chứa 0 (nền) và 255 (lá).")

    spot_mask = np.zeros(leaf_mask.shape, dtype=np.uint8)
    leaf_area = cv2.countNonZero(leaf_mask)
    if leaf_area == 0:
        return cv2.bitwise_and(spot_mask, leaf_mask)
    inside = leaf_mask == 255

    # 2. Chuyển BGR -> HSV và LAB. HSV uint8 của OpenCV có H trong [0,179],
    # S,V trong [0,255]. LAB uint8 có a=128 là trung tính, a thấp thiên xanh,
    # a cao thiên đỏ; b cao thiên vàng. Kết hợp hai hệ giúp hạn chế bắt nhầm
    # phần lá xanh tối chỉ vì độ sáng thấp.
    hsv = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2LAB)
    hue, saturation, value = cv2.split(hsv)
    lightness, a_channel, b_channel = cv2.split(lab)
    # Đổi float trước khi trừ để tránh tràn số uint8 (ví dụ 100 - 150).
    lightness_f = lightness.astype(np.float32)
    a_f = a_channel.astype(np.float32)

    # 3. Ước lượng màu tham chiếu CHỈ từ vùng lá. Nếu còn đủ mô xanh, dùng
    # median của mô xanh làm chuẩn, nhờ đó vẫn giữ được mảng hoại tử lớn.
    # Nếu lá chủ yếu vàng đồng (mặt dưới tự nhiên), dùng median toàn lá:
    # màu vàng/nâu đồng đồng đều không tự động bị coi là một đốm lớn.
    green = (
        inside & (hue >= 35) & (hue <= 95)
        & (saturation >= 40) & (value >= 45) & (a_channel < 128)
    )
    has_green_reference = np.count_nonzero(green) >= 0.10 * leaf_area
    reference = green if has_green_reference else inside
    reference_a = float(np.median(a_f[reference]))
    reference_l = float(np.median(lightness_f[reference]))

    # Trung bình cục bộ có trọng số mask: nền, kể cả nền đã tô đen, không
    # được kéo tối màu tham chiếu ở rìa lá. sigma tỉ lệ với căn diện tích lá
    # để cửa sổ lân cận thay đổi theo kích thước đối tượng.
    sigma = max(2.0, np.sqrt(leaf_area) * 0.025)
    weights = inside.astype(np.float32)
    weight_sum = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma)
    denominator = np.maximum(weight_sum, 1e-6)
    local_l = cv2.GaussianBlur(lightness_f * weights, (0, 0), sigmaX=sigma) / denominator
    local_a = cv2.GaussianBlur(a_f * weights, (0, 0), sigmaX=sigma) / denominator
    darker_than_neighbors = local_l - lightness_f
    redder_than_neighbors = a_f - local_a

    # 4. Ngưỡng màu cho ba nhóm đốm. Dải nâu bao gồm nâu đỏ; dải vàng
    # cho phép vùng vàng sáng, không chặn V ở 200 như phiên bản ban đầu.
    brown = cv2.inRange(hsv, (0, 50, 25), (25, 255, 255)) > 0
    yellow = cv2.inRange(hsv, (20, 65, 50), (38, 255, 255)) > 0
    if has_green_reference:
        brown &= a_f >= max(122.0, reference_a + 12.0)
        yellow &= (a_f >= max(105.0, reference_a + 12.0)) & (b_channel >= 145)
        warm_spots = brown | yellow
    else:
        # Trên lá vàng đồng, yêu cầu lệch đỏ so với màu nền của chính lá;
        # vùng lệch ít hơn phải đồng thời tối và đỏ hơn lân cận. Điều này
        # giảm bắt nhầm các gân sáng và các vùng đổi sáng nhẹ trên mặt dưới.
        abnormal_color = (a_f >= reference_a + 8.0) | (
            (a_f >= reference_a + 4.0)
            & (darker_than_neighbors >= 15.0) & (redder_than_neighbors >= 4.0)
        )
        warm_spots = (brown | yellow) & abnormal_color

    # Đốm đen có thể bão hòa thấp nên cần nhánh riêng, không phụ thuộc màu
    # nâu/vàng. Loại phần xanh tối thông thường; vùng gần đen vẫn được giữ
    # nếu tối hơn lân cận rõ rệt. Màu đen ở nền sẽ bị AND loại ngay sau đó.
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 55)) > 0
    not_green = (hue < 35) | (hue > 95) | (saturation < 45)
    dark_spots = dark & (
        (not_green & (reference_l - lightness_f >= 30.0))
        | ((value <= 25) & (darker_than_neighbors >= 12.0))
    )
    candidates = np.where(warm_spots | dark_spots, 255, 0).astype(np.uint8)
    candidates = cv2.bitwise_and(candidates, leaf_mask)

    # 5. Opening: erosion rồi dilation bằng ellipse 3x3, xóa nhiễu đơn lẻ
    # và các nét gân quá mảnh. Closing: dilation rồi erosion, nối khe đứt
    # và lấp lỗ nhỏ. Không lấp toàn bộ contour để tránh tăng diện tích giả.
    # AND sau mỗi phép toán vì dilation/closing có thể lan ra ngoài lá.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    candidates = cv2.bitwise_and(candidates, leaf_mask)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel)
    candidates = cv2.bitwise_and(candidates, leaf_mask)

    # 6. Lọc từng thành phần liên thông: bỏ chấm quá nhỏ theo diện tích lá.
    # Gân thường dài và mảnh; kiểm tra minAreaRect thay vì chỉ bounding box
    # trục ngang/dọc để phát hiện được cả gân chéo. Chỉ loại khi đồng thời
    # dài/mảnh, tránh xóa một vùng cháy lớn có hình dạng kéo dài.
    min_area = max(9, int(round(leaf_area * 0.00003)))
    thin_limit = max(4.0, np.sqrt(leaf_area) * 0.008)
    # Dải biên chỉ dùng để bỏ thành phần nhỏ hầu như bám sát rìa/cuống;
    # không erosion toàn lá, vì một vùng cháy lớn vẫn có thể chạm mép lá.
    edge_band = max(3.0, np.sqrt(leaf_area) * 0.025)
    padded_leaf = cv2.copyMakeBorder(leaf_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    edge_distance = cv2.distanceTransform(padded_leaf, cv2.DIST_L2, 5)[1:-1, 1:-1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area:
            continue
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        component_pixels = component > 0
        local_edge = edge_distance[y:y + height, x:x + width]
        edge_fraction = np.mean(local_edge[component_pixels] <= edge_band)
        if area < 0.03 * leaf_area and edge_fraction >= 0.90:
            continue
        points = cv2.findNonZero(component)
        rect_width, rect_height = cv2.minAreaRect(points)[1]
        short_side = min(rect_width, rect_height)
        long_side = max(rect_width, rect_height)
        elongation = long_side / max(short_side, 1.0)
        if elongation > 4.0:
            # Gân cong có hình chữ nhật bao khá rộng. Phân vị 75% bán kính
            # bên trong phản ánh độ mảnh tốt hơn chỉ dùng cạnh chữ nhật.
            padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
            radii = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
            typical_radius = float(np.percentile(radii[component_pixels], 75))
            if short_side <= thin_limit or typical_radius <= thin_limit:
                continue
        keep[label] = 255

    # 7. Ràng buộc bắt buộc tại bước trả về: mask đốm luôn nằm trọn trong
    # leaf_mask gốc, cùng (H,W), dtype uint8 và chỉ có hai giá trị 0/255.
    spot_mask = keep[labels]
    return cv2.bitwise_and(spot_mask, leaf_mask)
