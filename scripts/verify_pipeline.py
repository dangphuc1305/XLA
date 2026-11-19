import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from src.preprocessing import preprocess_image
from src.leaf_segmentation import segment_leaf
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity
from src.disease_classifier import classify_disease
from src.ai_model import diagnose_durian_leaf

def test_pipeline(img_path):
    img = cv2.imread(str(img_path))
    prep = preprocess_image(img)
    l_mask = segment_leaf(prep)
    s_mask = detect_spots(prep, l_mask)
    sev = calculate_severity(l_mask, s_mask)
    morph = classify_disease(prep, l_mask, s_mask, sev["severity_percentage"])
    ai_res = diagnose_durian_leaf(prep, morph_info=morph, metrics=sev)
    
    top1 = ai_res.get("top1", {})
    name = top1.get("name_vi")
    prob = top1.get("probability")
    code = top1.get("class_code")
    status = ai_res.get("status")
    print(f"[{img_path.name}] Status: {status} | Pred: {name} ({code}) - {prob:.1f}%")
    for p in ai_res.get("top_predictions", []):
        print(f"    - {p['name_vi']}: {p['probability']:.1f}%")

print("=== SPOTTED SAMPLES ===")
for p in sorted(Path("data/spotted").glob("*.png")):
    test_pipeline(p)

print("\n=== NORMAL SAMPLES ===")
for p in sorted(Path("data/normal").glob("*.png")):
    test_pipeline(p)
