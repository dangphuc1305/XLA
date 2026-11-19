import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from src.preprocessing import preprocess_image
from src.leaf_segmentation import segment_leaf
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity
from src.disease_classifier import classify_disease
from src.ai_model import DurianAIDiagnoser

diag = DurianAIDiagnoser.get_instance()
for name in ["hu1.png", "hu2.png", "hu3.png", "hu4.png", "hu5.png"]:
    img = cv2.imread(f"data/spotted/{name}")
    prep = preprocess_image(img)
    l_mask = segment_leaf(prep)
    s_mask = detect_spots(prep, l_mask)
    sev = calculate_severity(l_mask, s_mask)
    morph = classify_disease(prep, l_mask, s_mask, sev["severity_percentage"])
    ai = diag.predict(prep)
    print(f"=== {name} ===")
    print("Morph:", morph["disease_code"], morph["name_vi"], f"{morph['confidence']}%")
    preds = [f"{x['class_code']}: {x['probability']:.1f}%" for x in ai["top_predictions"]]
    print("AI Top3:", preds)
