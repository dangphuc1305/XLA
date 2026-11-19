"""Kiểm tra lưu PNG không mất dữ liệu và báo lỗi khi không thể ghi."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from main import process_image
from src.image_io import save_image
from src.preprocessing import load_image

ROOT = Path(__file__).resolve().parents[1]


class ImageIOTests(unittest.TestCase):
    def test_unicode_path_preserves_binary_pixels(self):
        mask = np.zeros((30, 40), np.uint8)
        mask[5:25, 8:32] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'đốm lá sầu riêng.png'
            save_image(path, mask)
            actual = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(actual, mask)

    def test_missing_parent_reports_write_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                save_image(Path(directory) / 'không tồn tại' / 'mask.png', np.zeros((10, 10), np.uint8))

    def test_cli_does_not_report_success_if_saving_fails(self):
        output, errors = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch('main.save_image', side_effect=OSError('Không ghi được ảnh')):
                with redirect_stdout(output), redirect_stderr(errors):
                    result = process_image(str(ROOT / 'data/spotted/hu1.png'), directory)
        self.assertIsNone(result)
        self.assertIn('LOI LUU ANH', errors.getvalue())
        self.assertNotIn('Saved', output.getvalue())

    def test_cli_writes_three_readable_results_in_unicode_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'kết quả lá sầu riêng'
            environment = os.environ.copy()
            environment['PYTHONIOENCODING'] = 'cp1252'
            completed = subprocess.run(
                [sys.executable, str(ROOT / 'main.py'),
                 str(ROOT / 'data/spotted/hu1.png'), '--output', str(target)],
                cwd=ROOT, env=environment, capture_output=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', errors='replace'))
            self.assertIn(str(target), completed.stdout.decode('utf-8'))
            files = sorted(target.glob('*.png'))
            self.assertEqual(len(files), 3)
            self.assertTrue(all(load_image(str(path)).size > 0 for path in files))


if __name__ == '__main__':
    unittest.main()
