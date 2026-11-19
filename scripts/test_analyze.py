import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from app import analyze

img_bytes = (ROOT / "data" / "test_user_crop.png").read_bytes()
res = analyze(img_bytes, "1.0.0")
processed_bgr, leaf_mask, spot_mask, annotated_bgr, metrics, morph_info, ai_result = res
print("=== ANALYZE FUNCTION TEST ON USER CROP LEAF ===")
print("Metrics:", metrics)
print("AI Top1:", ai_result["top1"]["name_vi"], f"{ai_result['top1']['probability']}%")
print("Top 3:", [(x["name_vi"], x["probability"]) for x in ai_result["top_predictions"]])
