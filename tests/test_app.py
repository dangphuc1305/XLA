"""Kiểm tra luồng giao diện thật: chọn ảnh, phân tích và đổi ảnh."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppFlowTests(unittest.TestCase):
    def test_sample_analysis_and_switching_do_not_show_stale_metrics(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        # Bộ ảnh hu1 đã kiểm tra: bảo đảm UI dùng đúng mask và tỷ lệ.
        self.assertEqual([m.value for m in app.metric], ['377,274', '16,188', '4.29%', 'Nhẹ'])
        self.assertEqual(len(app.get('image')), 4)
        app.selectbox[0].select(ROOT / 'data' / 'normal' / '3.png').run(timeout=30)
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[2].value, '0.12%')
        app.selectbox[0].select(ROOT / 'data' / 'normal' / '2.png').run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        # Nhánh bóng xanh rất tối trước đây tạo 20 pixel giả trên ảnh này.
        self.assertEqual(app.metric[2].value, '0.00%')
        self.assertEqual(app.metric[3].value, 'Bình thường')

    def test_upload_mode_without_file_disables_analysis(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        app.radio[0].set_value('Tải ảnh của bạn').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.metric), 0)


if __name__ == '__main__':
    unittest.main()
