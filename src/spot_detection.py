"""Phát hiện vùng đốm bất thường trên lá sầu riêng — phần Trần Minh Nhựt.

Quy ước: ảnh màu BGR uint8; mask một kênh uint8, chỉ chứa 0 và 255.
Đây là thuật toán ngưỡng màu và hình thái học, không chẩn đoán tác nhân bệnh.
"""

import cv2
import numpy as np


def _prune_boundary_tails(candidates: np.ndarray, edge_distance: np.ndarray,
                         band: int) -> np.ndarray:
    """Cắt nhánh mảnh sát biên dù nhánh đang nối với một mảng đốm lớn.

    Mask lá có thể lấn sang bóng đất sát mép. Lọc cả thành phần liên thông
    không giải quyết được khi bóng nối với vùng cháy thật. Thay vào đó,
    chỉ giữ phần sát biên được một lõi đốm đủ dày phía trong hỗ trợ ở gần.
    Không dùng phép tái dựng lan truyền đến hội tụ vì nó sẽ khôi phục lại
    cả nhánh bóng dài. Đốm nhỏ không có lõi như vậy được để lại cho bộ lọc
    diện tích/hình dạng hiện có, tránh xóa tất cả đốm nhỏ ở mép lá.
    """
    core_radius = max(1.0, band / 8.0)
    padded = cv2.copyMakeBorder(candidates, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    thickness = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    core = (candidates > 0) & (edge_distance > band) & (thickness >= core_radius)
    if not np.any(core):
        return candidates

    count, labels = cv2.connectedComponents(candidates, connectivity=8)
    has_core = np.zeros(count, dtype=bool)
    has_core[np.unique(labels[core])] = True
    # Chỉ cho lõi hỗ trợ nhánh sát biên trong một khoảng ngắn. Nếu dùng cả
    # chiều rộng dải biên, một lõi bệnh có thể giữ lại bóng mảnh chạy dọc
    # silhouette trong hàng chục pixel. Bán kính lõi giúp giữ mép vùng cháy.
    reach = max(4, band // 3) + int(np.ceil(core_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    supported = cv2.dilate(core.astype(np.uint8), kernel) > 0
    tails = (edge_distance <= band) & has_core[labels] & ~supported
    result = candidates.copy()
    result[tails] = 0
    return result


def _exclude_narrow_petiole(leaf_mask: np.ndarray, detection_mask: np.ndarray) -> np.ndarray:
    """Loại phần cuống hẹp ở một đầu trục dài khỏi ROI phát hiện.

    Cuống thường là phần nối dài hẹp hơn rõ rệt so với đầu nhọn còn lại của
    phiến. PCA tìm trục dài, sau đó so bề rộng hai dải đầu mút; chỉ cắt đầu
    mút hẹp bất đối xứng. Mask lá đầu vào không bị sửa.
    """
    y, x = np.where(leaf_mask > 0)
    if len(x) < 20:
        return detection_mask

    points = np.column_stack((x, y)).astype(np.float32)
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    projection = (points - center) @ axis
    across = (points - center) @ normal
    low, high = float(projection.min()), float(projection.max())
    length = high - low
    if length < 20:
        return detection_mask

    def robust_width(values: np.ndarray) -> float:
        if values.size < 10:
            return 0.0
        return float(np.percentile(values, 95) - np.percentile(values, 5))

    max_width = robust_width(across)
    band = length * 0.10
    low_width = robust_width(across[projection <= low + band])
    high_width = robust_width(across[projection >= high - band])
    narrow_end = "low" if low_width < high_width else "high"
    narrow_width = min(low_width, high_width)
    broad_width = max(low_width, high_width)
    # Nếu hai đầu gần tương đương, không đoán đầu nào là cuống.
    if max_width <= 0 or narrow_width >= 0.75 * broad_width or narrow_width >= 0.45 * max_width:
        return detection_mask

    step = max(4.0, length * 0.025)
    width_limit = max_width * 0.30
    trim_distance = 0.0
    for index in range(12):
        if narrow_end == "low":
            start, end = low + index * step, min(low + (index + 1) * step, high)
        else:
            start, end = max(high - (index + 1) * step, low), high - index * step
        if start >= end:
            break
        width = robust_width(across[(projection >= start) & (projection < end)])
        if width >= width_limit:
            break
        trim_distance = end - low if narrow_end == "low" else high - start

    if trim_distance == 0:
        return detection_mask
    # Thêm một bin để không còn pixel ứng viên ở phần chuyển tiếp cuống/phiến.
    trim_distance = min(length * 0.25, trim_distance + step)
    yy, xx = np.indices(leaf_mask.shape, dtype=np.float32)
    full_projection = (xx - center[0]) * axis[0] + (yy - center[1]) * axis[1]
    result = detection_mask.copy()
    if narrow_end == "low":
        result[full_projection <= low + trim_distance] = 0
    else:
        result[full_projection >= high - trim_distance] = 0
    return result


def _midvein_mask(leaf_mask: np.ndarray) -> np.ndarray:
    """Tạo dải hẹp quanh trục giữa để tránh nhận gân chính là đốm.

    PCA lấy trục dài của lá; dải bị loại chỉ rộng một phần nhỏ so với bề rộng
    tối đa của phiến. Dùng chủ yếu cho lá vàng đồng, nơi gân chính thường có
    màu nâu và rất dễ vượt ngưỡng màu.
    """
    y, x = np.where(leaf_mask > 0)
    if len(x) < 20:
        return np.zeros_like(leaf_mask)
    points = np.column_stack((x, y)).astype(np.float32)
    center = points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    axis = eigenvectors[:, -1]
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    along = (points - center) @ axis
    across = (points - center) @ normal
    width = float(np.percentile(across, 95) - np.percentile(across, 5))
    band = max(3.0, width * 0.04)
    low, high = float(along.min()), float(along.max())
    length = max(high - low, 1.0)
    bin_count = max(10, int(round(length / 40.0)))
    point_bins = np.minimum(((along - low) / length * bin_count).astype(np.int32), bin_count - 1)
    centers = np.full(bin_count, np.nan, dtype=np.float32)
    for index in range(bin_count):
        values = across[point_bins == index]
        if values.size:
            centers[index] = np.median(values)
    known = np.flatnonzero(np.isfinite(centers))
    if not known.size:
        return np.zeros_like(leaf_mask)
    centers = np.interp(np.arange(bin_count), known, centers[known]).astype(np.float32)
    yy, xx = np.indices(leaf_mask.shape, dtype=np.float32)
    full_along = (xx - center[0]) * axis[0] + (yy - center[1]) * axis[1]
    full_across = (xx - center[0]) * normal[0] + (yy - center[1]) * normal[1]
    full_bins = np.clip(((full_along - low) / length * bin_count).astype(np.int32), 0, bin_count - 1)
    centerline = centers[full_bins]
    return np.where(
        (leaf_mask > 0) & (np.abs(full_across - centerline) <= band), 255, 0
    ).astype(np.uint8)


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

    # Khoảng cách tới biên mask được tính sớm để nhận diện bóng sát mép lá.
    # Bóng đổ thường tạo một dải tối mảnh chạy dọc silhouette; vì nó có thể
    # nối với một vùng bệnh thật, chỉ lọc thành phần liên thông là chưa đủ.
    padded_leaf = cv2.copyMakeBorder(leaf_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    edge_distance = cv2.distanceTransform(padded_leaf, cv2.DIST_L2, 5)[1:-1, 1:-1]
    wide_edge_band = max(3, int(round(np.sqrt(leaf_area) * 0.055)))

    # Mở mask lá bằng ellipse vừa phải để tạo ROI cho phiến lá. Các nhánh
    # hẹp như cuống thường biến mất sau phép opening; mask lá gốc vẫn được
    # giữ nguyên và vẫn là ràng buộc cuối cho spot_mask.
    blade_kernel_size = max(5, int(round(np.sqrt(leaf_area) * 0.04)))
    if blade_kernel_size % 2 == 0:
        blade_kernel_size += 1
    blade_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blade_kernel_size, blade_kernel_size)
    )
    blade_detection_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, blade_kernel)
    # Với ảnh rất nhỏ hoặc mask lá rất mảnh, tránh để ROI rỗng.
    if cv2.countNonZero(blade_detection_mask) < 0.50 * leaf_area:
        blade_detection_mask = leaf_mask.copy()
    blade_detection_mask = _exclude_narrow_petiole(leaf_mask, blade_detection_mask)
    blade_inside = blade_detection_mask == 255

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
        blade_inside & (hue >= 35) & (hue <= 95)
        & (saturation >= 40) & (value >= 45) & (a_channel < 128)
    )
    has_green_reference = np.count_nonzero(green) >= 0.10 * leaf_area
    midvein = _midvein_mask(leaf_mask) > 0
    reference = green if has_green_reference else blade_inside
    reference_a = float(np.median(a_f[reference]))
    reference_l = float(np.median(lightness_f[reference]))

    # Trung bình cục bộ có trọng số mask: nền, kể cả nền đã tô đen, không
    # được kéo tối màu tham chiếu ở rìa lá. sigma tỉ lệ với căn diện tích lá
    # để cửa sổ lân cận thay đổi theo kích thước đối tượng.
    sigma = max(2.0, np.sqrt(leaf_area) * 0.025)
    weights = blade_inside.astype(np.float32)
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
        # Loại vùng vàng sáng phản quang/giọt nước: vàng bất thường phải có
        # tương phản cục bộ về sắc đỏ hoặc độ tối, không chỉ giống màu vàng.
        yellow &= (redder_than_neighbors >= 8.0) | (darker_than_neighbors >= 8.0)
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

        # Trên lá vàng đồng, gân giữa có thể ngả nâu giống bệnh. Bỏ riêng
        # dải hẹp quanh trục dài; các đốm lệch sang hai bên vẫn được xét.
        warm_spots &= ~midvein

    # Đốm đen có thể bão hòa thấp nên cần nhánh riêng, không phụ thuộc màu
    # nâu/vàng. Vùng còn giữ sắc xanh được xem là bóng trên mô xanh kể cả
    # khi V rất thấp; không nhận đốm chỉ vì tối hơn lân cận. Đốm đen trung
    # tính/nâu tối đi qua nhánh not_green. Nền bị AND loại ngay sau đó.
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 55)) > 0
    not_green = (hue < 35) | (hue > 95) | (saturation < 45)
    dark_spots = dark & not_green & (
        (reference_l - lightness_f >= 30.0)
        | ((value <= 25) & (darker_than_neighbors >= 12.0))
    )
    # Tâm hoại tử trên một số ảnh (đặc biệt hu4) có màu gần xám và sáng hơn
    # ngưỡng dark nghiêm ngặt ở trên, trong khi vòng nâu/vàng xung quanh đã
    # được nhận diện. Bổ sung phần lõi xám tối chỉ khi nó nằm sát ứng viên
    # nâu/vàng và tối rõ rệt so với lá/lân cận; không mở rộng nhánh này ra
    # toàn lá để tránh biến bóng đổ độc lập thành đốm bệnh.
    warm_neighborhood_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (35, 35)
    )
    near_warm_spot = cv2.dilate(
        warm_spots.astype(np.uint8), warm_neighborhood_kernel
    ) > 0
    dark_gray_cores = (
        (saturation <= 65) & (value <= 110)
        & (lightness_f <= reference_l - 35.0)
        & (darker_than_neighbors >= 25.0)
        & near_warm_spot
    )
    dark_spots |= dark_gray_cores
    if not has_green_reference:
        dark_spots &= ~midvein
    # Vùng tối sát silhouette có xác suất cao là bóng đổ/viền tối. Ở dải
    # biên, chỉ giữ pixel đồng thời có màu nâu/vàng bất thường; nhờ đó màu
    # bệnh thật vẫn được giữ qua warm_spots, còn nhánh đen thuần túy bị bỏ.
    near_leaf_edge = edge_distance <= wide_edge_band
    dark_spots &= ~near_leaf_edge | warm_spots
    # Pixel nâu/vàng bất thường sát biên vẫn được xét nếu nó thuộc leaf_mask.
    # Điều này giữ tổn thương chạm mép sau khi bộ tách lá khôi phục phần lõm
    # nông; vùng tối thuần ở mép vẫn bị hạn chế để không lấy bóng nền.
    edge_supported_spot = (
        (warm_spots | dark_gray_cores) & inside & near_leaf_edge
    )
    detection_roi = blade_inside | edge_supported_spot
    candidates = np.where(
        (warm_spots | dark_spots) & detection_roi, 255, 0
    ).astype(np.uint8)
    candidates = cv2.bitwise_and(candidates, leaf_mask)

    # 5. Opening: erosion rồi dilation bằng ellipse 3x3, xóa nhiễu đơn lẻ
    # và các nét gân quá mảnh. Closing: dilation rồi erosion, nối khe đứt
    # và lấp lỗ nhỏ. Không lấp toàn bộ contour để tránh tăng diện tích giả.
    # AND sau mỗi phép toán vì dilation/closing có thể lan ra ngoài lá.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    candidates = cv2.bitwise_and(candidates, blade_detection_mask)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel)
    candidates = cv2.bitwise_and(candidates, blade_detection_mask)

    # 6. Lọc từng thành phần liên thông: bỏ chấm quá nhỏ theo diện tích lá.
    # Gân thường dài và mảnh; kiểm tra minAreaRect thay vì chỉ bounding box
    # trục ngang/dọc để phát hiện được cả gân chéo. Chỉ loại khi đồng thời
    # dài/mảnh, tránh xóa một vùng cháy lớn có hình dạng kéo dài.
    # Chấm dưới vài chục pixel thường là gân/giọt nước/nhiễu sau tiền xử lý.
    min_area = max(48, int(round(leaf_area * 0.00008)))
    thin_limit = max(4.0, np.sqrt(leaf_area) * 0.008)
    # Dải biên chỉ dùng để bỏ thành phần nhỏ hầu như bám sát rìa/cuống;
    # không erosion toàn lá, vì một vùng cháy lớn vẫn có thể chạm mép lá.
    edge_band = max(3.0, np.sqrt(leaf_area) * 0.025)
    # Cắt nhánh bóng bám theo một mảng cháy lớn TRƯỚC khi lọc thành phần.
    # Sau khi cắt, các mảnh gân/cuống bị tách rời tiếp tục được lọc bên dưới.
    candidates = _prune_boundary_tails(candidates, edge_distance, wide_edge_band)
    candidates = cv2.bitwise_and(candidates, leaf_mask)
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
        wide_edge_fraction = np.mean(local_edge[component_pixels] <= wide_edge_band)
        local_midvein = midvein[y:y + height, x:x + width]
        midvein_fraction = np.mean(local_midvein[component_pixels])
        if area < max(min_area * 8, 0.0003 * leaf_area) and midvein_fraction >= 0.45:
            continue
        # Thành phần rất nhỏ bám biên thường là đầu cuống/nhánh nền lọt vào
        # leaf_mask; dùng ngưỡng edge rộng hơn cho nhóm nhiễu này.
        if area < 0.0035 * leaf_area and edge_fraction >= 0.70:
            continue
        if area < 0.03 * leaf_area and edge_fraction >= 0.85:
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
            # Khi bóng tách khỏi mảng cháy (ví dụ đổi độ sáng), nó có thể
            # thành một dải cong độc lập. Dải dài, mảnh, gần như toàn bộ
            # bám viền cũng bị loại dù chữ nhật bao của nó khá rộng.
            is_border_strip = wide_edge_fraction >= 0.90 and typical_radius <= wide_edge_band / 2.0
            if short_side <= thin_limit or typical_radius <= thin_limit or is_border_strip:
                continue
        keep[label] = 255

    # 7. Ràng buộc bắt buộc tại bước trả về: mask đốm luôn nằm trọn trong
    # leaf_mask gốc, cùng (H,W), dtype uint8 và chỉ có hai giá trị 0/255.
    spot_mask = keep[labels]
    # Ảnh hu4 có lõi hoại tử xám đen nằm kín bên trong một vòng ứng viên
    # nâu/vàng. Nếu lõi hơi sáng hoặc lẫn màu, ngưỡng màu có thể để lại lỗ
    # đen dù vùng bao quanh đã được nhận diện. Chỉ lấp lỗ kín khi phần bên
    # trong là lõi xám đen, tối rõ so với nền lá, và lỗ không chiếm phần lớn
    # thành phần; lỗ xanh/sáng vẫn được giữ nguyên.
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        spot_mask, connectivity=8
    )
    for label in range(1, component_count):
        component_area = int(component_stats[label, cv2.CC_STAT_AREA])
        component = np.where(component_labels == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue
        for hole_index, contour in enumerate(contours):
            if hierarchy[0, hole_index, 3] < 0:
                continue
            hole_area = float(cv2.contourArea(contour))
            if hole_area <= 0 or hole_area >= 0.5 * component_area:
                continue
            hole = np.zeros_like(spot_mask)
            cv2.drawContours(hole, [contour], -1, 255, thickness=cv2.FILLED)
            hole_pixels = (hole > 0) & (component_labels != label) & inside
            if not np.any(hole_pixels):
                continue
            hole_border = cv2.dilate(
                hole,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) > 0
            ring_pixels = hole_border & (component_labels == label)
            if not np.any(ring_pixels):
                continue
            hole_lightness = float(np.median(lightness_f[hole_pixels]))
            hole_saturation = float(np.median(saturation[hole_pixels]))
            if hole_lightness <= reference_l - 35.0 and hole_saturation <= 65.0:
                spot_mask[hole_pixels] = 255
    return cv2.bitwise_and(spot_mask, leaf_mask)
