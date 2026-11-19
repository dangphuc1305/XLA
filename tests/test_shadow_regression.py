"""Hồi quy trên ảnh hu2 do người dùng chỉ ra lỗi bóng đổ sát mép lá."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from src.preprocessing import load_image, preprocess_image
from src.leaf_segmentation import segment_leaf
from src.spot_detection import detect_spots

ROOT = Path(__file__).resolve().parents[1]


class ShadowRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image = preprocess_image(load_image(str(ROOT / 'data/spotted/hu2.png')))
        cls.image = image
        cv2.setRNGSeed(0)
        cls.leaf = segment_leaf(image)
        cls.spots = detect_spots(image, cls.leaf)

    def test_lower_healthy_edge_and_cast_shadow_are_not_spots(self):
        # ROI đối chiếu ảnh gốc: mép dưới xanh + bóng trên đất, không gồm
        # vùng cháy nâu nằm ở nửa trên/phải. Bản trước có 8.687 pixel sai.
        self.assertEqual(np.count_nonzero(self.spots[850:1030, 200:700]), 0)

    def test_brown_necrosis_is_retained_and_mask_stays_inside_leaf(self):
        # Các điểm nằm trong mô cháy nâu quan sát được trên ảnh gốc.
        for y, x in ((250, 1000), (400, 1000), (650, 800)):
            self.assertEqual(self.spots[y, x], 255)
        self.assertGreater(np.count_nonzero(self.spots[200:500, 900:1100]), 25000)
        self.assertFalse(np.any(self.spots[self.leaf == 0]))

    def test_border_shadow_stays_rejected_after_brightness_changes(self):
        distance = cv2.distanceTransform(self.leaf, cv2.DIST_L2, 5)
        shadow_region = np.zeros(self.leaf.shape, bool)
        shadow_region[850:1030, 200:700] = True
        shadow_region &= (self.leaf > 0) & (distance <= 41)
        for factor in (0.75, 1.25):
            with self.subTest(brightness=factor):
                changed = np.clip(self.image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                spots = detect_spots(changed, self.leaf)
                self.assertEqual(np.count_nonzero(spots[shadow_region]), 0)
                self.assertEqual(spots[400, 1000], 255)


if __name__ == '__main__':
    unittest.main()
