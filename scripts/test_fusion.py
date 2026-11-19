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
from src.ai_model import DurianAIDiagnoser, CLASS_NAMES

MORPH_TO_CLASS = {
    "algal_spot": "Leaf_Algal",
    "anthracnose": "Leaf_Colletotrichum",
    "healthy": "Leaf_Healthy",
    "phomopsis": "Leaf_Phomopsis",
    "rhizoctonia": "Leaf_Rhizoctonia",
}

def fuse_predictions(probs: np.ndarray, morph_info: dict, metrics: dict):
    fused = probs.copy()
    
    # 1. Nếu hoàn toàn không có đốm bệnh
    if metrics and (metrics.get("spot_area_px", 0) == 0 or metrics.get("severity_percentage", 0.0) <= 0.03):
        h_idx = CLASS_NAMES.index("Leaf_Healthy")
        fused = np.zeros_like(probs)
        fused[h_idx] = 1.0
        return fused
        
    # 2. Nếu có đặc trưng hình thái học mạnh
    if morph_info and morph_info.get("disease_code") in MORPH_TO_CLASS:
        target_cls = MORPH_TO_CLASS[morph_info["disease_code"]]
        t_idx = CLASS_NAMES.index(target_cls)
        conf = float(morph_info.get("confidence", 85.0)) / 100.0
        
        # Nếu là Phomopsis (nhiều đốm mắt cua có quầng vàng)
        if morph_info["disease_code"] == "phomopsis":
            morph_prior = np.zeros_like(probs)
            morph_prior[t_idx] = conf
            fused = 0.35 * probs + 0.65 * morph_prior
            fused = fused / fused.sum()
        elif morph_info["disease_code"] == "rhizoctonia" and fused[CLASS_NAMES.index("Leaf_Rhizoctonia")] > 0.40:
            # Tăng cường Rhizoctonia khi cả 2 cùng chỉ báo
            fused[CLASS_NAMES.index("Leaf_Rhizoctonia")] += 0.30
            fused = fused / fused.sum()
            
    return fused

diag = DurianAIDiagnoser.get_instance()

print("--- KIỂM TRA TẬP SPOTTED ---")
for p in sorted(Path("data/spotted").glob("*.png")):
    img = cv2.imread(str(p))
    prep = preprocess_image(img)
    l_mask = segment_leaf(prep)
    s_mask = detect_spots(prep, l_mask)
    sev = calculate_severity(l_mask, s_mask)
    morph = classify_disease(prep, l_mask, s_mask, sev["severity_percentage"])
    
    # Raw AI probs
    inp = diag._onnx_session.run(None, {"input": diag.predict(prep)["metadata"] if False else None}) if False else None
    res = diag.predict(prep)
    print(f"File: {p.name}")
    print(f"  Morph: {morph['name_vi']} ({morph['confidence']}%)")
    print(f"  AI raw: {res.get('top1', {}).get('name_vi')} ({res.get('top1', {}).get('probability')}%)")
