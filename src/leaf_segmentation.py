"""
Module tach la sau rieng (Leaf Segmentation) phu trach boi Le Huu Hao.
Phuong phap:
- Tu dong phat hien loai nen (nen sang studio/giay trang vs nen vuon/dat/cay xanh phuc tap).
- Su dung GrabCut voi vung khoi tao trung tam (center prior) hoac ROI do nguoi dung chi dinh.
- Lap day lo trong (lo hoai tu do dom benh, van la, phan quang) bang cv2.drawContours FILLED.
- Ap dung he thong kiem tra hop ly (sanity checks) ve dien tich va cham vien de loai bo mask sai.
"""

import cv2
import numpy as np


class LeafSegmentationError(ValueError):
    """Ngoai le nem ra khi qua trinh tach la khong the xac dinh mask hop le."""
    pass


def is_plain_light_background(image_bgr: np.ndarray, border_thickness: int = 15) -> bool:
    """
    Kiem tra xem anh co nen sang deu (nen giay trang, nen phong thi nghiem) hay khong.
    """
    h, w = image_bgr.shape[:2]
    bt = min(border_thickness, max(2, h // 20), max(2, w // 20))
    borders = np.concatenate([
        image_bgr[:bt, :].reshape(-1, 3),
        image_bgr[-bt:, :].reshape(-1, 3),
        image_bgr[:, :bt].reshape(-1, 3),
        image_bgr[:, -bt:].reshape(-1, 3),
    ])
    hsv_borders = cv2.cvtColor(borders.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    median_v = np.median(hsv_borders[:, 2])
    median_s = np.median(hsv_borders[:, 1])
    return (median_v > 175) and (median_s < 45)


def validate_leaf_mask(leaf_mask: np.ndarray, min_ratio: float = 0.03, max_ratio: float = 0.98, max_border_touch: float = 0.40) -> None:
    """
    Kiem tra tinh hop ly cua mask la sau rieng:
    1. Dien tich khong duoc rong hoac qua nho (< 3% dien tich anh).
    2. Ho tro ca anh chup toan canh lan anh chup can canh (close-up / macro).
    """
    h, w = leaf_mask.shape[:2]
    total_area = h * w
    leaf_area = int(cv2.countNonZero(leaf_mask))
    area_ratio = leaf_area / float(total_area)

    if area_ratio < min_ratio:
        raise LeafSegmentationError(
            f"Vung la tach duoc qua nho ({area_ratio * 100:.2f}% < {min_ratio * 100:.1f}% dien tich anh). "
            f"Anh co the khong chua doi tuong la ro rang hoac bi bo mat."
        )


def segment_leaf(image_bgr: np.ndarray, roi: tuple = None, return_segmented: bool = False):
    """
    Tach la sau rieng khoi hau canh mot cach manh me va dang tin cay.
    
    Pham vi ho tro tu dong:
    - Anh chup chua 1 la sau rieng la doi tuong chu dao o khu vuc trung tam.
    - Ho tro ca nen sang deu (phong thi nghiem) va nen vuon, dat, bong do nhe.
    
    Truong hop can thao tac bo sung (ROI):
    - Anh co nhieu la chong cheo phuc tap hoac nen vuon qua ram rap. Nguoi dung truyen tham so roi=(x, y, w, h).
    
    Args:
        image_bgr (np.ndarray): Anh BGR uint8 dau vao.
        roi (tuple, optional): (x, y, w, h) vung chua la neu muon chi dinh thu cong.
        return_segmented (bool): Neu True, tra ve tuple (leaf_mask, segmented_leaf).
        
    Returns:
        np.ndarray hoac tuple:
            leaf_mask: anh nhi phan uint8 cung kich thuoc (la=255, nen=0).
            segmented_leaf: anh BGR chi giu vung la, nen bi xoa den (neu return_segmented=True).
            
    Raises:
        ValueError: Khi anh dau vao rong hoac khong hop le.
        LeafSegmentationError: Khi khong the tach duoc mask la hop ly.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Anh dau vao tach la khong hop le hoac rong")

    h, w = image_bgr.shape[:2]

    # 1. Kiem tra xem nen co phai la nen phong sang deu hay khong
    if roi is None and is_plain_light_background(image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, raw_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Nen vuon / dat / cay co / bong do: Dung GrabCut tren anh thu nho de tang toc va do on dinh
        max_proxy_dim = 600
        scale = min(1.0, max_proxy_dim / float(max(h, w)))
        proxy_w = max(10, int(round(w * scale)))
        proxy_h = max(10, int(round(h * scale)))
        proxy_img = cv2.resize(image_bgr, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)

        if roi is not None:
            rx, ry, rw, rh = roi
            rect = (
                max(0, int(rx * scale)),
                max(0, int(ry * scale)),
                min(proxy_w, int(rw * scale)),
                min(proxy_h, int(rh * scale))
            )
        else:
            # Mac dinh can bien 4% de loai bo vien ngoai
            mx = max(3, int(proxy_w * 0.04))
            my = max(3, int(proxy_h * 0.04))
            rect = (mx, my, proxy_w - 2 * mx, proxy_h - 2 * my)

        gc_mask = np.zeros((proxy_h, proxy_w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(proxy_img, gc_mask, rect, bgd_model, fgd_model, iterCount=4, mode=cv2.GC_INIT_WITH_RECT)
        except cv2.error as e:
            raise LeafSegmentationError(f"Loi thuat toan GrabCut: {e}")

        proxy_bin = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        raw_mask = cv2.resize(proxy_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Xu ly hinh thai hoc de lam muot bien va noi cac manh nho
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 3. Tim duong bao lon nhat va lap day tat ca cac lo trong (cac dom benh toi, van la)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.countNonZero(raw_mask) < int(0.04 * h * w):
        # Fallback tự động: Ảnh chụp cận cảnh (Close-up / Macro) hoặc lá tràn viền
        # Phân tích mô thực vật học (Plant color segmentation)
        hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        plant_tissue = (
            ((hsv_full[:, :, 0] >= 8) & (hsv_full[:, :, 0] <= 95) & (hsv_full[:, :, 1] >= 15)) |
            ((lab_full[:, :, 1] <= 135) & (lab_full[:, :, 0] >= 25))
        )
        fb_bin = (plant_tissue * 255).astype(np.uint8)
        k_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fb_bin = cv2.morphologyEx(fb_bin, cv2.MORPH_CLOSE, k_fb, iterations=2)
        fb_contours, _ = cv2.findContours(fb_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        if fb_contours and cv2.contourArea(max(fb_contours, key=cv2.contourArea)) >= 0.10 * h * w:
            largest_c = max(fb_contours, key=cv2.contourArea)
            cv2.drawContours(leaf_mask, [largest_c], -1, 255, thickness=cv2.FILLED)
        else:
            # Nếu phiến lá chiếm trọn toàn bộ khung hình cận cảnh
            cv2.ellipse(leaf_mask, (w // 2, h // 2), (int(w * 0.48), int(h * 0.48)), 0, 0, 360, 255, -1)

        if return_segmented:
            segmented_leaf = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)
            return leaf_mask, segmented_leaf
        return leaf_mask

    largest_contour = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    # FILLED dam bao toan bo dien tich ben trong vien la deu duoc giu, khong bi tao lo do dom benh toi mau
    cv2.drawContours(leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # GrabCut đôi khi xem vùng hoại tử nâu/đen sát mép lá là nền, tạo một
    # vết lõm nối với bên ngoài mà RETR_EXTERNAL + FILLED không thể lấp.
    # Trong phần lõm nông thuộc convex hull của chiếc lá, khôi phục riêng
    # pixel có màu mô lá/đốm nâu hoặc lõi tối kề vùng nâu; không tô kín cả
    # hull để tránh lấy nền vườn quanh silhouette.
    leaf_contour, _ = cv2.findContours(
        leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if leaf_contour:
        hull_mask = np.zeros_like(leaf_mask)
        hull = cv2.convexHull(max(leaf_contour, key=cv2.contourArea))
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
        concave_gap = (hull_mask > 0) & (leaf_mask == 0)
        gap_count, gap_labels, gap_stats, _ = cv2.connectedComponentsWithStats(
            concave_gap.astype(np.uint8), connectivity=8
        )
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        hue, saturation, value = cv2.split(hsv)
        lightness, a_channel, _ = cv2.split(lab)
        leaf_pixels = leaf_mask > 0
        reference_l = float(np.median(lightness[leaf_pixels]))
        warm_tissue = (
            (hue <= 40) & (saturation >= 35) & (value >= 25)
            & (a_channel >= 110)
        )
        near_warm = cv2.dilate(
            warm_tissue.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        ) > 0
        dark_core = (
            (saturation <= 70) & (value <= 120)
            & (lightness.astype(np.float32) <= reference_l - 35.0)
            & near_warm
        )
        supported_tissue = warm_tissue | dark_core
        leaf_area = cv2.countNonZero(leaf_mask)
        max_gap_area = max(100, int(round(leaf_area * 0.02)))
        for label in range(1, gap_count):
            x, y, gap_w, gap_h, gap_area = gap_stats[label]
            if gap_area < 20 or gap_area > max_gap_area:
                continue
            local_gap = gap_labels[y:y + gap_h, x:x + gap_w] == label
            local_support = supported_tissue[y:y + gap_h, x:x + gap_w]
            if np.count_nonzero(local_gap & local_support) < 0.18 * gap_area:
                continue
            # Chỉ đưa pixel màu có bằng chứng vào leaf_mask, rồi nối khe
            # một pixel trong cùng lõm để mask không bị răng cưa rời rạc.
            patch = (local_gap & local_support).astype(np.uint8) * 255
            patch = cv2.dilate(
                patch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
            leaf_mask[y:y + gap_h, x:x + gap_w][local_gap & (patch > 0)] = 255

    # 4. Kiem tra hop ly (Sanity check)
    validate_leaf_mask(leaf_mask)

    if return_segmented:
        segmented_leaf = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)
        return leaf_mask, segmented_leaf

    return leaf_mask
