"""Kiểm thử hợp đồng mask và các tình huống lỗi của thuật toán Nhựt.

Chạy từ thư mục dự án: python -m unittest discover -s tests -v
Ảnh tổng hợp có vị trí đốm biết trước; không thay cho đánh giá trên ảnh thật.
"""

import unittest

import cv2
import numpy as np

from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity


class DetectSpotsTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((180, 240, 3), (20, 100, 180), np.uint8)
        self.leaf = np.zeros((180, 240), np.uint8)
        self.leaf[15:165, 20:220] = 255
        self.image[self.leaf == 255] = (40, 150, 40)  # BGR xanh lá.

    def assert_valid(self, mask):
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, self.leaf.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all((mask == 0) | (mask == 255)))
        self.assertFalse(np.any(mask[self.leaf == 0]))

    def test_background_brown_and_healthy_leaf_are_not_spots(self):
        result = detect_spots(self.image, self.leaf)
        self.assert_valid(result)
        self.assertEqual(cv2.countNonZero(result), 0)

    def test_brown_yellow_and_black_disks(self):
        for x, color in ((60, (30, 75, 140)), (120, (0, 255, 255)), (180, (12, 12, 12))):
            cv2.circle(self.image, (x, 85), 13, color, -1)
        result = detect_spots(self.image, self.leaf)
        self.assert_valid(result)
        for x in (60, 120, 180):
            self.assertEqual(result[85, x], 255)
            self.assertGreater(np.count_nonzero(result[77:94, x - 8:x + 9]), 250)

    def test_spot_crossing_leaf_edge_and_mask_hole_stays_inside(self):
        cv2.circle(self.image, (20, 80), 18, (30, 75, 140), -1)
        cv2.circle(self.image, (120, 80), 18, (30, 75, 140), -1)
        self.leaf[78:83, 118:123] = 0
        result = detect_spots(self.image, self.leaf)
        self.assert_valid(result)
        self.assertEqual(result[80, 25], 255)
        self.assertFalse(np.any(result[78:83, 118:123]))

    def test_single_pixel_noise_and_thin_vein_removed(self):
        self.image[40, 45] = (30, 75, 140)
        cv2.line(self.image, (80, 30), (80, 145), (30, 75, 140), 2)
        cv2.line(self.image, (110, 30), (200, 140), (30, 75, 140), 2)
        result = detect_spots(self.image, self.leaf)
        self.assertEqual(cv2.countNonZero(result), 0)

    def test_closing_fills_small_hole_in_spot(self):
        cv2.circle(self.image, (120, 85), 15, (30, 75, 140), -1)
        self.image[85, 120] = (40, 150, 40)
        result = detect_spots(self.image, self.leaf)
        self.assertEqual(result[85, 120], 255)

    def test_large_necrotic_region_is_retained(self):
        self.image[20:160, 25:175] = (30, 75, 140)
        result = detect_spots(self.image, self.leaf)
        self.assertGreater(np.count_nonzero(result), 20000)

    def test_uniform_bronze_leaf_is_not_a_large_spot(self):
        self.image[self.leaf > 0] = (65, 140, 180)
        result = detect_spots(self.image, self.leaf)
        self.assertEqual(cv2.countNonZero(result), 0)

    def test_thick_elongated_lesion_is_not_removed_as_vein(self):
        self.image[70:100, 40:200] = (30, 75, 140)
        result = detect_spots(self.image, self.leaf)
        self.assertGreater(np.count_nonzero(result), 4500)

    def test_noncontiguous_input_and_full_leaf_mask(self):
        self.leaf[:] = 255
        cv2.circle(self.image, (120, 85), 13, (30, 75, 140), -1)
        expected = detect_spots(self.image, self.leaf)
        actual = detect_spots(self.image[:, ::-1], self.leaf[:, ::-1])
        np.testing.assert_array_equal(actual[:, ::-1], expected)
        self.assert_valid(actual)

    def test_brown_spot_on_bronze_leaf(self):
        self.image[self.leaf > 0] = (65, 140, 180)
        cv2.circle(self.image, (120, 85), 13, (30, 75, 140), -1)
        result = detect_spots(self.image, self.leaf)
        self.assertEqual(result[85, 120], 255)

    def test_dark_green_shadow_is_not_a_black_spot(self):
        self.image[40:140, 60:180] = (12, 48, 12)
        result = detect_spots(self.image, self.leaf)
        self.assertEqual(cv2.countNonZero(result), 0)

    def test_nearly_black_green_shadow_keeps_its_green_chromaticity(self):
        # Trước khi sửa: nhánh V<=25 bỏ qua sắc xanh và đánh dấu cả bóng.
        self.image[40:140, 60:180] = (4, 18, 4)
        cv2.circle(self.image, (190, 85), 10, (12, 12, 12), -1)
        result = detect_spots(self.image, self.leaf)
        self.assertFalse(np.any(result[50:130, 70:170]))
        self.assertEqual(result[85, 190], 255)  # Đốm đen thật vẫn được giữ.

    def test_boundary_shadow_attached_to_large_lesion_is_pruned(self):
        # Tăng kích thước để có dải bóng đủ rộng sống sót qua Opening 3x3.
        image = np.full((400, 500, 3), (40, 150, 40), np.uint8)
        leaf = np.zeros((400, 500), np.uint8)
        leaf[20:380, 20:480] = 255
        cv2.rectangle(image, (380, 230), (470, 379), (30, 75, 140), -1)
        # Nhánh nâu tối dài nằm sát mép, nối trực tiếp vào vùng cháy.
        cv2.rectangle(image, (70, 367), (400, 379), (20, 45, 75), -1)
        result = detect_spots(image, leaf)
        self.assertFalse(np.any(result[368:378, 90:300]))
        self.assertEqual(result[300, 430], 255)
        self.assertEqual(result[374, 430], 255)  # Giữ vùng cháy thật chạm mép.

    def test_empty_leaf_and_tiny_image(self):
        self.leaf[:] = 0
        result = detect_spots(self.image, self.leaf)
        self.assert_valid(result)
        self.assertEqual(cv2.countNonZero(result), 0)
        tiny = detect_spots(np.full((1, 1, 3), 100, np.uint8), np.full((1, 1), 255, np.uint8))
        self.assertEqual(tiny.shape, (1, 1))

    def test_inputs_unchanged_and_black_background_equivalent(self):
        cv2.circle(self.image, (120, 85), 13, (30, 75, 140), -1)
        before_image, before_leaf = self.image.copy(), self.leaf.copy()
        first = detect_spots(self.image, self.leaf)
        masked = cv2.bitwise_and(self.image, self.image, mask=self.leaf)
        second = detect_spots(masked, self.leaf)
        np.testing.assert_array_equal(self.image, before_image)
        np.testing.assert_array_equal(self.leaf, before_leaf)
        np.testing.assert_array_equal(first, second)

    def test_invalid_inputs_raise_clear_errors(self):
        invalid = [
            (None, self.leaf, TypeError),
            (self.image.astype(np.float32), self.leaf, TypeError),
            (self.image, self.leaf.astype(bool), TypeError),
            (self.image[:, :, 0], self.leaf, ValueError),
            (self.image[:, :, :2], self.leaf, ValueError),
            (np.zeros((0, 0, 3), np.uint8), np.zeros((0, 0), np.uint8), ValueError),
            (self.image, self.leaf[:100], ValueError),
            (self.image, self.leaf[:, :, None], ValueError),
            (self.image, np.ones_like(self.leaf), ValueError),
        ]
        for image, leaf, exception in invalid:
            with self.subTest(shape=getattr(image, 'shape', None), error=exception):
                with self.assertRaises(exception):
                    detect_spots(image, leaf)

    def test_phuc_receives_masks_and_computes_area_ratio(self):
        cv2.circle(self.image, (120, 85), 13, (30, 75, 140), -1)
        spots = detect_spots(self.image, self.leaf)
        metrics = calculate_severity(self.leaf, spots)
        expected = round(np.count_nonzero(spots) / np.count_nonzero(self.leaf) * 100, 2)
        self.assertEqual(metrics['severity_percentage'], expected)
        self.assertLessEqual(metrics['spot_area_px'], metrics['leaf_area_px'])


if __name__ == '__main__':
    unittest.main()
