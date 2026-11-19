import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from src.ai_model import diagnose_durian_leaf

test_dir = ROOT / "dataset" / "mendeley_durian" / "Durian_Leaf_Diseases_Release_20261017_v2" / "test"
print("--- TESTING MENDELEY TEST SET SAMPLES ---")
for c_dir in sorted(test_dir.iterdir()):
    if c_dir.is_dir():
        imgs = sorted(list(c_dir.glob("*.jpg")))[:3]
        for p in imgs:
            img = cv2.imread(str(p))
            res = diagnose_durian_leaf(img)
            status = res.get("status")
            if status != "SUCCESS":
                print(f"⚠️ {c_dir.name} [{p.name}] REJECTED/ERROR: {res.get('message')}")
                continue
            top1 = res.get("top1", {})
            cls_code = top1.get("class_code", "None")
            prob = top1.get("probability", 0.0)
            name = top1.get("name_vi", "Unknown")
            match = "✅" if cls_code == c_dir.name else "❌"
            print(f"{match} True: {c_dir.name:<18} | Pred: {cls_code:<18} ({prob:.1f}%) -> {name}")
