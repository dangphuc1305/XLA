import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from src.ai_model import diagnose_durian_leaf

print("--- TESTING DATA/SPOTTED ---")
for p in sorted(Path("data/spotted").glob("*.png")):
    img = cv2.imread(str(p))
    res = diagnose_durian_leaf(img)
    top1 = res.get("top1", {})
    name = top1.get("name_vi")
    prob = top1.get("probability", 0.0)
    preds = [f"{x['name_vi']}: {x['probability']:.1f}%" for x in res.get("top_predictions", [])]
    print(f"{p.name} -> Top1: {name} ({prob:.1f}%) | Top3: {preds}")

print("\n--- TESTING DATA/NORMAL ---")
for p in sorted(Path("data/normal").glob("*.png")):
    img = cv2.imread(str(p))
    res = diagnose_durian_leaf(img)
    top1 = res.get("top1", {})
    name = top1.get("name_vi")
    prob = top1.get("probability", 0.0)
    preds = [f"{x['name_vi']}: {x['probability']:.1f}%" for x in res.get("top_predictions", [])]
    print(f"{p.name} -> Top1: {name} ({prob:.1f}%) | Top3: {preds}")
