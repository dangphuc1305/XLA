import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from src.preprocessing import preprocess_image
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity
from src.disease_classifier import classify_disease
from src.ai_model import diagnose_durian_leaf

img = cv2.imread("data/test_user_crop.png")
prep = preprocess_image(img)
h, w = prep.shape[:2]

# Close-up fallback mask
leaf_mask = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(leaf_mask, (w // 2, h // 2), (int(w * 0.48), int(h * 0.48)), 0, 0, 360, 255, -1)

spot_mask = detect_spots(prep, leaf_mask)
metrics = calculate_severity(leaf_mask, spot_mask)
morph_info = classify_disease(prep, leaf_mask, spot_mask, metrics["severity_percentage"])
ai_res = diagnose_durian_leaf(prep, morph_info=morph_info, metrics=metrics)

print("Full pipeline on user image SUCCESS!")
print("Metrics:", metrics)
print("Morph:", morph_info["name_vi"], morph_info["confidence"])
print("AI Top1:", ai_res["top1"]["name_vi"], f"{ai_res['top1']['probability']}%")
print("Top 3:", [(x["name_vi"], x["probability"]) for x in ai_res["top_predictions"]])
