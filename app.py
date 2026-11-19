"""
🍃 HỆ THỐNG CHẨN ĐOÁN SÂU BỆNH TRÊN LÁ SẦU RIÊNG BẰNG THỊ GIÁC MÁY TÍNH & AI DEEP LEARNING
Giao diện 3D Dark Mode chuẩn công thái học, trực quan, tối ưu tương phản.
Tích hợp Hệ Thống Máy Học Tăng Cường (Active Learning & Long-term Memory Bank).
Nhóm tác giả: Lê Hữu Hào, Trần Minh Nhựt, Đặng Hoàng Phúc.
Mô hình AI: MobileNetV3-Small (Transfer Learning - Mendeley Durian Leaf Dataset v5).
"""

import hashlib
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import streamlit as st

from src.image_io import encode_image
from src.preprocessing import preprocess_image
from src.leaf_segmentation import segment_leaf, LeafSegmentationError
from src.spot_detection import detect_spots
from src.severity_analysis import calculate_severity
from src.disease_classifier import classify_disease
from src.ai_model import (
    diagnose_durian_leaf,
    check_image_quality,
    EXTENDED_DISEASE_ENCYCLOPEDIA,
)
from src.database import save_diagnosis, get_recent_history, clear_history
from src.active_memory import (
    get_all_learned_diseases,
    save_learned_disease,
    delete_learned_disease,
    search_treatment_and_medicine,
    seed_initial_learned_diseases,
    extract_visual_embedding,
    find_matching_learned_disease,
)

ROOT = Path(__file__).resolve().parent

# Khóa phiên bản thuật toán gắn với toàn bộ mã nguồn xử lý và phân loại
ALGO_FILES = [
    ROOT / "src" / "preprocessing.py",
    ROOT / "src" / "leaf_segmentation.py",
    ROOT / "src" / "spot_detection.py",
    ROOT / "src" / "severity_analysis.py",
    ROOT / "src" / "disease_classifier.py",
    ROOT / "src" / "ai_model.py",
    ROOT / "src" / "database.py",
    ROOT / "src" / "active_memory.py",
]
DETECTOR_VERSION = hashlib.sha256(
    "".join(f.read_text(encoding="utf-8") for f in ALGO_FILES if f.exists()).encode("utf-8")
).hexdigest()[:16]

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Lá Sầu Riêng | AI & Computer Vision",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- TỐI ƯU GIAO DIỆN 3D DARK MODE HIỆN ĐẠI (UI/UX 3D ENGINE) -----------------
st.markdown("""
<style>
    /* Bảng màu Dark Mode chuẩn: Slate & Emerald Theme */
    .stApp {
        background-color: #080D14 !important;
        color: #F1F5F9 !important;
    }
    
    /* Header Hero 3D hiện đại Ultra-Cockpit */
    .hero-banner-3d {
        background: radial-gradient(130% 140% at 10% 0%, #064E3B 0%, #062E25 40%, #031410 85%, #020B09 100%);
        border: 1px solid rgba(52, 211, 153, 0.38);
        border-top: 1px solid rgba(167, 243, 208, 0.55);
        border-radius: 22px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 24px 60px -12px rgba(4, 120, 87, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.25), inset 0 -2px 6px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    .hero-banner-3d::before {
        content: "";
        position: absolute;
        top: -80px;
        right: -60px;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(52, 211, 153, 0.08) 50%, transparent 70%);
        pointer-events: none;
        filter: blur(24px);
    }
    .hero-banner-3d::after {
        content: "";
        position: absolute;
        bottom: -60px;
        left: 20%;
        width: 260px;
        height: 180px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, transparent 70%);
        pointer-events: none;
        filter: blur(30px);
    }
    .hero-emblem-3d {
        width: 60px;
        height: 60px;
        min-width: 60px;
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.35) 0%, rgba(6, 78, 59, 0.7) 100%);
        border: 1.5px solid rgba(52, 211, 153, 0.6);
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.05rem;
        box-shadow: 0 10px 24px rgba(16, 185, 129, 0.38), inset 0 2px 3px rgba(255, 255, 255, 0.45);
        transform: perspective(600px) rotateY(-8deg) rotateX(4deg);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .hero-emblem-3d:hover {
        transform: perspective(600px) rotateY(0deg) rotateX(0deg) scale(1.06);
    }
    .hero-title-gradient {
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -0.6px;
        margin: 0 0 5px 0;
        line-height: 1.25;
        background: linear-gradient(90deg, #FFFFFF 0%, #D1FAE5 45%, #6EE7B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
    }
    .hero-subtitle-tech {
        font-size: 0.95rem;
        color: #A7F3D0;
        line-height: 1.55;
        font-weight: 500;
        letter-spacing: 0.2px;
        margin: 0;
    }
    .hero-pill-badge {
        background: linear-gradient(145deg, rgba(22, 34, 53, 0.92) 0%, rgba(13, 21, 34, 0.96) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 9px 16px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-size: 0.83rem;
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.18);
        transition: all 0.25s ease;
        cursor: default;
    }
    .hero-pill-badge:hover {
        transform: translateY(-3px) scale(1.03);
        border-color: rgba(52, 211, 153, 0.6);
        box-shadow: 0 10px 24px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    /* HIỆU ỨNG CHỌN 3D (3D SEGMENTED PILL SELECTION) CHO RADIO OPTIONS */
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        background: #0F172A !important;
        padding: 6px 8px !important;
        border-radius: 14px !important;
        border: 1px solid #1E293B !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.5) !important;
        gap: 8px !important;
        width: fit-content !important;
    }
    div[role="radiogroup"] > label {
        background: #151F30 !important;
        border: 1px solid #233148 !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        margin: 0 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        cursor: pointer !important;
    }
    div[role="radiogroup"] > label:hover {
        transform: translateY(-2px) !important;
        border-color: #10B981 !important;
        box-shadow: 0 6px 14px rgba(16, 185, 129, 0.25) !important;
    }
    div[role="radiogroup"] > label:has(input:checked),
    div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        border-color: #34D399 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    }

    /* 4 THẺ CHỈ SỐ ĐỊNH LƯỢNG NỔI 3D (3D METRIC CARDS) */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #162235 0%, #0E1624 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        box-shadow: 0 10px 25px -4px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.015) !important;
        border-color: #10B981 !important;
        box-shadow: 0 18px 32px -4px rgba(16, 185, 129, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.6) !important;
    }

    /* NÚT BẤM 3D CÔNG THÁI HỌC VỚI HIỆU ỨNG NHẤN CHÌM TACTILE */
    div.stButton > button {
        background: linear-gradient(180deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #34D39966 !important;
        border-radius: 30px !important;
        padding: 11px 32px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 5px 0 #047857, 0 12px 22px rgba(16, 185, 129, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 7px 0 #047857, 0 16px 28px rgba(16, 185, 129, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
    }
    div.stButton > button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 2px 0 #047857, 0 4px 10px rgba(16, 185, 129, 0.3) !important;
    }
    div.stButton > button:disabled {
        background: #1E293B !important;
        color: #64748B !important;
        border-color: #334155 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        transform: none !important;
    }

    /* THẺ CHẨN ĐOÁN 3D NỔI BẬT */
    .card-3d {
        background: linear-gradient(145deg, #131E2F 0%, #0D1522 100%);
        border: 1px solid #233148;
        border-radius: 16px;
        padding: 22px 26px;
        margin: 16px 0;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .card-3d:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.14);
    }

    /* HUY HIỆU 3D BỘ NHỚ MÁY HỌC (GOLDEN AMBER 3D BADGE) */
    .badge-learned-3d {
        background: linear-gradient(135deg, #B45309 0%, #F59E0B 100%);
        color: #FFFFFF !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35);
        display: inline-flex !important;
        align-items: center;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        gap: 6px;
    }

    /* HUY HIỆU 3D DEEP LEARNING (EMERALD 3D BADGE) */
    .badge-ai-3d {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35);
        display: inline-flex !important;
        align-items: center;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        gap: 6px;
    }

    /* KHUNG DẠY MÁY HỌC 3D (ACTIVE LEARNING CONTAINER) */
    .learn-box-3d {
        background: linear-gradient(145deg, #171B38 0%, #0F1326 100%);
        border: 1px solid #4338CA66;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 12px 28px rgba(67, 56, 202, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* 4 THẺ THÔNG SỐ TRIỆU CHỨNG HÌNH THÁI HỌC */
    .symp-card {
        background: linear-gradient(145deg, #131E30 0%, #0D1524 100%);
        border: 1px solid #1E2D44;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .symp-card:hover {
        transform: translateY(-2px);
        border-color: #10B981;
    }
    .symp-header {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .symp-body {
        font-size: 0.90rem;
        font-weight: 600;
        color: #F1F5F9;
        line-height: 1.35;
    }

    /* KHUNG COCKPIT ĐẦU VÀO 3D */
    .cockpit-box-3d {
        background: linear-gradient(145deg, #101826 0%, #0A0F1A 100%);
        border: 1px solid #1E293B;
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SIDEBAR ĐIỀU KHIỂN & THÔNG TIN ĐỒ ÁN -----------------
with st.sidebar:
    st.markdown("### 🍃 Durian AI & CV Core")
    st.caption("Thị giác máy tính & Deep Learning chẩn đoán bệnh lá sầu riêng")
    
    st.markdown("---")
    st.markdown("#### 👥 Nhóm Sinh Viên Thực Hiện")
    st.markdown("""
    - **Lê Hữu Hào** *(Nhóm trưởng)*: Tiền xử lý & Phân đoạn GrabCut
    - **Trần Minh Nhựt**: Phát hiện đốm bất thường HSV & LAB
    - **Đặng Hoàng Phúc**: Định lượng tổn thương, UI 3D & Active Learning AI
    """)
    
    st.markdown("---")
    st.markdown("#### 🧠 Động Cơ Trí Tuệ Nhân Tạo")
    st.markdown("""
    - **Kiến trúc cốt lõi:** MobileNetV3-Small (Transfer Learning)
    - **Độ chính xác (Test):** `88.43%`
    - **Bộ nhớ máy học tăng cường:** `Active Learning Few-Shot Vector Memory`
    - **Số bệnh trong bộ nhớ:** `{count}` bệnh
    """.format(count=len(get_all_learned_diseases())))

    st.markdown("---")
    st.markdown("#### 🎯 Thang Cảnh Báo Mức Độ Bệnh")
    st.markdown("""
    - 🟢 **Bình thường** (0%): Sạch bệnh
    - 🟡 **Nhẹ** (>0% – <5%): Vết đốm chớm phát sinh
    - 🟠 **Trung bình** (5% – <20%): Đốm hoại tử lan tỏa
    - 🔴 **Nặng** (≥20%): Tổn thương nghiêm trọng
    """)
    
    st.markdown("---")
    st.caption(f"Engine: OpenCV `{cv2.__version__}` · ONNX Runtime")


# ----------------- BANNER HEADER HERO 3D -----------------
st.markdown(
    '<div class="hero-banner-3d">'
    '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">'
    '<div style="display: flex; align-items: center; gap: 20px;">'
    '<div class="hero-emblem-3d"><span>🍃</span></div>'
    '<div>'
    '<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">'
    '<span style="background: rgba(16, 185, 129, 0.22); border: 1px solid rgba(52, 211, 153, 0.5); border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; color: #6EE7B7; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase;">'
    '🟢 AI VISION 2.0 · LIVE INFERENCE'
    '</span>'
    '<span style="color: #64748B; font-size: 0.78rem;">|</span>'
    '<span style="color: #94A3B8; font-size: 0.78rem; font-weight: 600;">Đồ án Xử Lý Ảnh Chuyên Sâu</span>'
    '</div>'
    '<h1 class="hero-title-gradient">Hệ Thống Chẩn Đoán Bệnh Lá Sầu Riêng</h1>'
    '<p class="hero-subtitle-tech">'
    'Thị giác máy tính 4 bước <span style="color: #38BDF8; font-weight: 700;">GrabCut & HSV/LAB</span> · AI Phân loại <span style="color: #FCD34D; font-weight: 700;">MobileNetV3</span> · Bộ Nhớ Tăng Cường <span style="color: #F472B6; font-weight: 700;">Active Learning</span>'
    '</p>'
    '</div>'
    '</div>'
    '<div style="display: flex; gap: 10px; flex-wrap: wrap;">'
    '<div class="hero-pill-badge" style="border-left: 3.5px solid #10B981;">'
    '<span style="font-size: 1.15rem;">🧠</span>'
    '<div>'
    '<div style="color: #F1F5F9; font-size: 0.85rem; font-weight: 800; line-height: 1.2;">MobileNetV3</div>'
    '<div style="color: #10B981; font-size: 0.70rem; font-weight: 700;">ACC: 88.43%</div>'
    '</div>'
    '</div>'
    '<div class="hero-pill-badge" style="border-left: 3.5px solid #38BDF8;">'
    '<span style="font-size: 1.15rem;">🔬</span>'
    '<div>'
    '<div style="color: #F1F5F9; font-size: 0.85rem; font-weight: 800; line-height: 1.2;">OpenCV 4.x</div>'
    '<div style="color: #38BDF8; font-size: 0.70rem; font-weight: 700;">GrabCut / HSV</div>'
    '</div>'
    '</div>'
    '<div class="hero-pill-badge" style="border-left: 3.5px solid #F59E0B;">'
    '<span style="font-size: 1.15rem;">💾</span>'
    '<div>'
    '<div style="color: #F1F5F9; font-size: 0.85rem; font-weight: 800; line-height: 1.2;">Active Memory</div>'
    '<div style="color: #F59E0B; font-size: 0.70rem; font-weight: 700;">48D Vector Hub</div>'
    '</div>'
    '</div>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


def analyze(image_bytes, detector_version):
    """Pipeline phân tích ảnh kết hợp Thị giác máy tính 4 bước, AI Deep Learning và Bộ nhớ máy học."""
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Không đọc được ảnh. Hãy chọn tệp JPG, PNG hoặc WebP hợp lệ.")
    
    # 1. 4 bước Thị giác máy tính thuần túy
    processed_bgr = preprocess_image(image_bgr)
    cv2.setRNGSeed(0)
    try:
        leaf_mask = segment_leaf(processed_bgr)
    except Exception:
        # Fallback an toàn tự động cho ảnh cận cảnh (Close-up Macro) hoặc lá tràn viền
        h_p, w_p = processed_bgr.shape[:2]
        leaf_mask = np.zeros((h_p, w_p), dtype=np.uint8)
        cv2.ellipse(leaf_mask, (w_p // 2, h_p // 2), (int(w_p * 0.48), int(h_p * 0.48)), 0, 0, 360, 255, -1)

    spot_mask = detect_spots(processed_bgr, leaf_mask)
    metrics = calculate_severity(leaf_mask, spot_mask)
    
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_bgr = processed_bgr.copy()
    cv2.drawContours(annotated_bgr, contours, -1, (0, 0, 255), 2)
    
    # 2. Phân tích hình thái học vi thể
    morph_info = classify_disease(
        processed_bgr, leaf_mask, spot_mask, metrics["severity_percentage"]
    )
    
    # 3. Chẩn đoán tác nhân bệnh bằng AI Deep Learning kết hợp Thị giác máy tính & Bộ nhớ máy học
    ai_result = diagnose_durian_leaf(processed_bgr, morph_info=morph_info, metrics=metrics, leaf_mask=leaf_mask)
    
    return processed_bgr, leaf_mask, spot_mask, annotated_bgr, metrics, morph_info, ai_result


# ----------------- KHU VỰC ĐIỀU KHIỂN & TẢI ẢNH (BỐ CỤC 3D CÂN ĐỐI) -----------------
with st.container():
    col_input1, col_input2 = st.columns([1.18, 0.82], gap="large")
    
    with col_input1:
        st.markdown("##### 📥 1. Chọn nguồn ảnh lá sầu riêng:")
        source = st.radio(
            "Nguồn ảnh", 
            ["Ảnh mẫu kiểm thử", "Tải ảnh từ máy", "Chụp từ Camera"], 
            horizontal=True,
            label_visibility="collapsed"
        )
        
        image_bytes, image_name = None, None
        
        if source == "Ảnh mẫu kiểm thử":
            sample_options = []
            
            # Ảnh gốc đồ án
            project_spotted = sorted((ROOT / "data" / "spotted").glob("*.png"))
            project_normal = sorted((ROOT / "data" / "normal").glob("*.png"))
            for p in project_spotted:
                sample_options.append((p, f"🔴 [Mẫu đồ án] Đốm bệnh · {p.name}"))
            for p in project_normal:
                sample_options.append((p, f"🟢 [Mẫu đồ án] Bình thường · {p.name}"))
                
            # Ảnh mẫu thực tế từ bộ dữ liệu Mendeley
            mendeley_samples = sorted((ROOT / "data" / "mendeley_samples").glob("*.jpg"))
            for p in mendeley_samples:
                name_clean = p.name.replace("Leaf_", "")
                sample_options.append((p, f"🌾 [Mendeley Thực Tế] {name_clean}"))
                
            if sample_options:
                selected_sample = st.selectbox(
                    "Chọn ảnh mẫu kiểm thử:",
                    sample_options,
                    format_func=lambda x: x[1]
                )
                try:
                    image_bytes = selected_sample[0].read_bytes()
                    image_name = selected_sample[0].stem
                except OSError as exc:
                    st.error(f"Không đọc được ảnh mẫu: {exc}")
                    
        elif source == "Tải ảnh từ máy":
            uploaded = st.file_uploader("Tải tệp ảnh lá sầu riêng (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                image_bytes = uploaded.getvalue()
                image_name = Path(uploaded.name).stem
                
        else: # Chụp từ camera
            camera_pic = st.camera_input("Chụp ảnh trực tiếp từ camera thiết bị:")
            if camera_pic is not None:
                image_bytes = camera_pic.getvalue()
                image_name = f"camera_{camera_pic.name}"

        st.markdown(
            '<div style="background: #0D1522; border: 1px dashed #233148; border-radius: 12px; padding: 10px 14px; margin-top: 14px; font-size: 0.84rem; color: #94A3B8; line-height: 1.45;">'
            '💡 <b>Mẹo ảnh chuẩn:</b> Chụp phiến lá đơn cự ly gần (cách 20–30cm), bề mặt lá hướng về phía camera, đủ ánh sáng tự nhiên để thuật toán GrabCut và MobileNetV3 đạt độ chính xác cao nhất.'
            '</div>',
            unsafe_allow_html=True
        )

    with col_input2:
        st.markdown("##### 👁️ 2. Xem trước & Kiểm định chất lượng:")
        if image_bytes:
            try:
                preview_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if preview_bgr is not None:
                    # Kiểm tra chất lượng ảnh (Gatekeeper)
                    is_ok, quality_msg, meta = check_image_quality(preview_bgr)
                    
                    st.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), caption=f"Mẫu: {image_name}", width=240)
                    
                    if not is_ok:
                        st.warning(f"⚠️ {quality_msg}")
                    else:
                        st.markdown(
                            f'<div style="background: #0D1624; border: 1px solid #1E2E44; border-radius: 10px; padding: 6px 12px; font-size: 0.8rem; color: #10B981; font-weight: 600; text-align: center; margin-top: 6px;">'
                            f'✅ Ảnh đạt chuẩn: {preview_bgr.shape[1]}x{preview_bgr.shape[0]} px | Độ nét: {meta.get("blur_score", 0)}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(f"Lỗi đọc preview: {e}")
        else:
            st.markdown(
                '<div style="background: #090E17; border: 1px dashed #223249; border-radius: 14px; padding: 40px 16px; text-align: center; color: #64748B;">'
                '<span style="font-size: 2.2rem;">🍃</span>'
                '<div style="margin-top: 8px; font-weight: 700; color: #94A3B8; font-size: 0.95rem;">Chưa có ảnh nào được chọn</div>'
                '<div style="font-size: 0.82rem; margin-top: 4px; color: #64748B;">Vui lòng chọn ảnh mẫu hoặc tải ảnh lá từ máy</div>'
                '</div>',
                unsafe_allow_html=True
            )

# Bố cục nút bấm 3D cân đối ở giữa
st.write("")
col_b1, col_b2, col_b3 = st.columns([1, 1.8, 1])
with col_b2:
    analyze_btn = st.button("🚀 Bắt đầu phân tích & Chẩn đoán", type="primary", disabled=not image_bytes, use_container_width=True)

# Quản lý session state
image_key = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
current_key = (source, image_key, DETECTOR_VERSION)
if st.session_state.get("input_key") != current_key:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("teach_lookup_result", None)
    st.session_state["input_key"] = current_key

if analyze_btn:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("teach_lookup_result", None)
    try:
        with st.spinner("Đang tách phiến lá GrabCut, trích xuất đặc trưng và so khớp bộ nhớ AI..."):
            st.session_state["analysis_result"] = analyze(image_bytes, DETECTOR_VERSION)
    except LeafSegmentationError as exc:
        st.error(f"Lỗi tách phiến lá: {exc}")
        st.info("Khắc phục: Chọn ảnh chụp một phiến lá đơn ở trung tâm, phông nền đơn sắc hoặc thoáng sáng.")
    except (ValueError, OSError, cv2.error) as exc:
        st.error(f"Lỗi phân tích: {exc}")

result = st.session_state.get("analysis_result")

# ----------------- KHU VỰC HIỂN THỊ KẾT QUẢ GỌN GÀNG, CHUYÊN NGHIỆP -----------------
if result is not None:
    processed_bgr, leaf_mask, spot_mask, annotated_bgr, metrics, morph_info, ai_result = result
    
    grade_labels = {
        "Healthy (Normal)": "Bình thường", 
        "Low / Early Stage": "Nhẹ",
        "Moderate Stage": "Trung bình", 
        "Severe Stage": "Nặng",
    }
    
    st.markdown("---")
    
    # =========================================================================
    # PHẦN 1: QUY TRÌNH 4 BƯỚC THỊ GIÁC MÁY TÍNH (HIỂN THỊ Ở TRÊN THEO YÊU CẦU)
    # =========================================================================
    st.markdown(f"#### 📸 Quy trình 4 bước phân tích thị giác máy tính: **{image_name}**")
    img_cols = st.columns(4)
    for col, title, img, is_bgr in zip(
        img_cols,
        ("1. Ảnh đã tiền xử lý", "2. Mask lá", "3. Mask đốm", "4. Đốm khoanh đỏ"),
        (processed_bgr, leaf_mask, spot_mask, annotated_bgr),
        (True, False, False, True),
    ):
        with col:
            st.subheader(title)
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if is_bgr else img
            st.image(disp, width="stretch")

    st.markdown("---")
    
    # =========================================================================
    # PHẦN 2: THỐNG KÊ ĐỊNH LƯỢNG TỔN THƯƠNG (4 THẺ NỔI 3D DARK MODE)
    # =========================================================================
    st.markdown("#### 📊 Thống kê số liệu & Định lượng tổn thương phiến lá:")
    
    cols = st.columns(4)
    cols[0].metric("Diện tích lá (pixel)", f"{metrics['leaf_area_px']:,}")
    cols[1].metric("Diện tích đốm (pixel)", f"{metrics['spot_area_px']:,}")
    percent_label = (
        "<0.01%" if metrics["spot_area_px"] > 0 and metrics["severity_percentage"] == 0
        else f"{metrics['severity_percentage']:.2f}%"
    )
    cols[2].metric("Tỷ lệ diện tích đốm", percent_label)
    cols[3].metric("Mức theo ngưỡng", grade_labels[metrics["grade"]])

    # =========================================================================
    # PHẦN 3: KẾT QUẢ CHẨN ĐOÁN BỆNH BẰNG AI & BỘ NHỚ MÁY HỌC (3D CARD)
    # =========================================================================
    if st.session_state.get("just_learned_msg"):
        st.balloons()
        st.success(st.session_state.pop("just_learned_msg"))

    st.markdown("#### 🧠 Kết quả chẩn đoán bệnh học:")
    
    top1 = ai_result.get("top1")
    top_preds = ai_result.get("top_predictions", [])
    ai_status = ai_result.get("status", "SUCCESS")
    conf_label = ai_result.get("confidence_label", "Độ tin cậy xác định")
    is_learned = ai_result.get("is_learned", False)
    
    if top1:
        disease_name_vi = top1["name_vi"]
        scientific_name = top1["scientific_name"]
        prob = top1["probability"]
        cls_code = top1["class_code"]

        # Tự động lưu vào cơ sở dữ liệu SQLite
        try:
            save_diagnosis(
                image_name=image_name,
                disease_code=cls_code,
                disease_name_vi=disease_name_vi,
                scientific_name=scientific_name,
                ai_probability=prob,
                severity_percentage=metrics["severity_percentage"],
                severity_grade=metrics["grade"],
                model_version="Active-Memory-v2.0" if is_learned else "MobileNetV3-Durian-v1.0",
                status=ai_status
            )
        except Exception:
            pass

        # Màu sắc viền & badge theo trạng thái
        if is_learned:
            card_accent = "#F59E0B" # Màu vàng kim (Học máy)
            badge_html = f'<span class="badge-learned-3d">🎓 ĐÃ NHẬN DIỆN TỪ BỘ NHỚ ({prob:.1f}%)</span>'
        elif "Khỏe Mạnh" in disease_name_vi:
            card_accent = "#10B981" # Xanh lục
            badge_html = f'<span class="badge-ai-3d">🟢 PHIẾN LÁ KHỎE MẠNH ({prob:.1f}%)</span>'
        elif prob >= 75.0:
            card_accent = "#10B981" # Xanh lục chuẩn
            badge_html = f'<span class="badge-ai-3d">🎯 XÁC SUẤT AI: {prob:.1f}%</span>'
        else:
            card_accent = "#3B82F6" # Xanh dương
            badge_html = f'<span class="badge-ai-3d" style="background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%);">🔍 XÁC SUẤT AI: {prob:.1f}%</span>'

        # BỐ CỤC CHẨN ĐOÁN 3D
        col_res1, col_res2 = st.columns([1.35, 1.05], gap="medium")
        
        with col_res1:
            teacher_tag = f"<div style='color: #FCD34D; font-size: 0.88rem; margin: 4px 0;'>👨‍🌾 <b>Ghi nhận bởi:</b> {top1.get('teacher_name', 'Chuyên gia')}</div>" if is_learned else ""
            symptoms_escaped = top1.get('symptoms', '').replace('"', '&quot;')
            cause_escaped = top1.get('cause', '').replace('"', '&quot;')

            card_html = (
                f'<div class="card-3d" style="border-left: 6px solid {card_accent}; margin-bottom: 12px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">'
                f'{badge_html}'
                f'<span style="color: #64748B; font-size: 0.82rem; font-family: monospace;">MÃ LỚP: {cls_code}</span>'
                f'</div>'
                f'<h2 style="color: #FFFFFF; margin: 6px 0 2px 0; font-size: 1.7rem; font-weight: 800;">{disease_name_vi}</h2>'
                f'<div style="color: #94A3B8; font-style: italic; font-size: 1.0rem; margin-bottom: 6px;">Tên khoa học: {scientific_name}</div>'
                f'{teacher_tag}'
                f'<div style="color: #60A5FA; font-size: 0.86rem; font-weight: 600; margin-bottom: 8px;">Độ tin cậy: {conf_label}</div>'
                f'<div style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.5; margin: 6px 0;"><b>Triệu chứng nhận dạng:</b> {symptoms_escaped}</div>'
                f'<div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.4;"><b>Nguyên nhân & điều kiện:</b> {cause_escaped}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

        with col_res2:
            st.markdown("##### 📈 Tỷ lệ dự đoán & Độ tin cậy (Top-3):")
            for p in top_preds:
                p_name = p["name_vi"]
                p_prob = p["probability"]
                is_p_learned = p.get("is_learned", False)
                prefix = "🎓 [Đã học] " if is_p_learned else ""
                st.write(f"**{prefix}{p_name}**: `{p_prob:.1f}%`")
                st.progress(min(1.0, max(0.0, p_prob / 100.0)))

        # Phác đồ điều trị và thuốc bảo vệ thực vật 3D Card
        active_str = ", ".join(top1.get("active_ingredients", [])) or "Tham khảo chuyên gia bảo vệ thực vật"
        med_str = ", ".join(top1.get("medicines", [])) or "Tham khảo danh mục thuốc BVTV được phép sử dụng"
        
        treatment_html = (
            f'<div class="card-3d" style="padding: 16px 20px; margin-top: 10px; border-left: 4px solid #10B981;">'
            f'<div style="color: #10B981; font-weight: 700; font-size: 0.94rem; margin-bottom: 4px;">💊 Hoạt chất khuyến cáo bảo vệ thực vật:</div>'
            f'<div style="color: #F1F5F9; font-size: 0.92rem; margin-bottom: 10px; font-weight: 500;">{active_str}</div>'
            f'<div style="color: #10B981; font-weight: 700; font-size: 0.94rem; margin-bottom: 4px;">🛒 Thuốc thương phẩm tiêu biểu trên thị trường:</div>'
            f'<div style="color: #F1F5F9; font-size: 0.92rem; font-weight: 500;">{med_str}</div>'
            f'</div>'
        )
        st.markdown(treatment_html, unsafe_allow_html=True)

        with st.expander("📋 Xem chi tiết phác đồ canh tác & Luận cứ hình thái học", expanded=False):
            st.markdown("##### Các bước xử lý đề xuất:")
            for step in top1.get("protocol", []):
                st.markdown(f"- {step}")
            if top1.get("notes"):
                st.info(f"💡 **Lưu ý chuyên môn:** {top1['notes']}")
                
            st.markdown("---")
            st.markdown("##### 🔬 Đối chiếu định lượng hình thái học Computer Vision:")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write(f"📍 **Vị trí phân bố:** {morph_info.get('spatial_desc', 'Toàn phiến lá')}")
            with col_m2:
                st.write(f"🔬 **Dạng vết tổn thương:** {morph_info.get('morphology_desc', 'Đốm hoại tử')}")
            if morph_info.get("name_vi") == disease_name_vi:
                ev_text = morph_info.get("evidence", "")
            else:
                ev_text = f"Đặc trưng hình thái vết thương: {morph_info.get('morphology_desc')}, phân bố tại {morph_info.get('spatial_desc')}, tổng diện tích tổn thương chiếm {metrics['severity_percentage']:.2f}% phiến lá. Mô hình AI MobileNetV3 xác định tác nhân chính là {disease_name_vi}."
            st.caption(f"Luận cứ chẩn đoán: {ev_text}")

    # =========================================================================
    # PHẦN 4: HỆ THỐNG MÁY HỌC TĂNG CƯỜNG (ACTIVE LEARNING & TRÍ THỨC MÁY HỌC)
    # CHỈ HIỂN THỊ KHI BỆNH CHƯA CÓ TRONG HỆ THỐNG HOẶC ĐỘ TIN CẬY THẤP (< 70%)
    # KHI ĐÃ BIẾT RỒI THÌ ẨN ĐI, KHÔNG CẦN HỌC NỮA
    # =========================================================================
    is_known_disease = is_learned or (prob >= 70.0)

    if not is_known_disease:
        st.markdown("---")
        st.markdown(
            f'<div style="background: linear-gradient(135deg, #2D1B00 0%, #1E1300 100%); border: 1px solid #F59E0B66; border-left: 6px solid #F59E0B; border-radius: 14px; padding: 16px 20px; margin: 16px 0; box-shadow: 0 10px 25px rgba(245, 158, 11, 0.15);">'
            f'<div style="color: #FBBF24; font-weight: 800; font-size: 1.02rem; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">'
            f'<span>⚠️</span> PHÁT HIỆN TRIỆU CHỨNG LẠ / ĐỘ TIN CẬY DƯỚI NGƯỠNG AN TOÀN ({prob:.1f}%)'
            f'</div>'
            f'<div style="color: #E2E8F0; font-size: 0.90rem; line-height: 1.5;">'
            f'Hình ảnh này có đặc trưng tổn thương chưa có trong 6 lớp bệnh chuẩn hoặc AI chưa thể khẳng định chắc chắn. '
            f'Hệ thống tự động kích hoạt chức năng <b>Dạy Máy Học (Active Learning)</b> để bạn nạp tên bệnh thực tế, '
            f'tự động tra cứu hoạt chất & thuốc đặc trị và ghi nhớ vĩnh viễn vào Bộ Nhớ Máy Học.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        with st.expander("💡 Dạy máy học / Nạp tên bệnh mới & Tự động tìm kiếm thuốc trị", expanded=True):
            st.markdown(
                '<div class="learn-box-3d">'
                '<div style="font-size: 1.15rem; font-weight: 800; color: #C7D2FE; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">'
                '<span>🧠</span> TRUNG TÂM NẠP TRI THỨC BỆNH MỚI (ACTIVE LEARNING ENGINE)'
                '</div>'
                '<div style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.5;">'
                'Hệ thống kết hợp <b>Thị giác máy tính trích xuất vector đặc trưng 48 chiều</b> và <b>Từ điển nông nghiệp chuyên sâu</b>. '
                'Chỉ cần xác nhận tên bệnh thực tế, AI sẽ tự động phân tích thành phần dược tính, tra cứu hoạt chất tối ưu và lưu vào cơ sở tri thức để tự động nhận diện chính xác ở các lần chụp sau!'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            st.write("")
            st.markdown("##### 🔬 BƯỚC 1: Đặc trưng hình thái học máy tính quan sát được trên lá:")
            c_sym1, c_sym2, c_sym3, c_sym4 = st.columns(4)
            with c_sym1:
                st.markdown(
                    f'<div class="symp-card">'
                    f'<div class="symp-header"><span>🔬</span> Dạng tổn thương</div>'
                    f'<div class="symp-body">{morph_info.get("morphology_desc", "Đốm hoại tử")}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c_sym2:
                st.markdown(
                    f'<div class="symp-card">'
                    f'<div class="symp-header"><span>📍</span> Vị trí phân bố</div>'
                    f'<div class="symp-body">{morph_info.get("spatial_desc", "Toàn phiến lá")}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c_sym3:
                st.markdown(
                    f'<div class="symp-card">'
                    f'<div class="symp-header"><span>📊</span> Tỷ lệ tổn thương</div>'
                    f'<div class="symp-body">{metrics["severity_percentage"]:.2f}% ({grade_labels[metrics["grade"]]})</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c_sym4:
                st.markdown(
                    f'<div class="symp-card">'
                    f'<div class="symp-header"><span>🎨</span> Sắc tố màu sắc</div>'
                    f'<div class="symp-body">Nâu rỉ sắt & hoại tử viền (HSV/LAB)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.write("")
            st.markdown("##### 📝 BƯỚC 2: Nạp tên bệnh thực tế & Tra cứu thuốc đặc trị:")
            
            entered_name = st.text_input(
                "Tên bệnh hoặc tác nhân dịch hại thực tế của cây sầu riêng:",
                placeholder="Ví dụ: Đốm Lá Do Vi Khuẩn, Rệp Sáp, Cháy Chóp Lá, Rầy Nhảy, Nứt Thân Xì Mủ...",
                key="input_teach_disease"
            )

            col_btn_l1, col_btn_l2, col_btn_l3 = st.columns([1, 2, 1])
            with col_btn_l2:
                do_lookup = st.button("🔍 Tự động tra cứu hoạt chất, thuốc trị & phác đồ", type="primary", use_container_width=True)

            if do_lookup and entered_name.strip():
                with st.spinner(f"Đang phân tích từ điển nông nghiệp tra cứu phác đồ cho '{entered_name}'..."):
                    v_sym = f"Vết thương dạng {morph_info.get('morphology_desc')} phân bố ở {morph_info.get('spatial_desc')}, tỷ lệ tổn thương {metrics['severity_percentage']:.2f}%."
                    lookup_res = search_treatment_and_medicine(entered_name, visual_symptoms=v_sym)
                    st.session_state["teach_lookup_data"] = lookup_res
                    st.success(f"✅ Đã tìm thấy hoạt chất & phác đồ tối ưu cho '{entered_name}'!")
            elif do_lookup and not entered_name.strip():
                st.warning("Vui lòng chọn hoặc nhập tên bệnh trước khi tra cứu!")

            lookup_data = st.session_state.get("teach_lookup_data")
            if lookup_data:
                st.markdown("---")
                st.markdown(f"#### 💊 BƯỚC 3: Phác đồ đề xuất & Xác nhận ghi nhớ cho: **{lookup_data.get('disease_name', entered_name)}**")
                
                c_f1, c_f2 = st.columns(2, gap="medium")
                with c_f1:
                    t_sci = st.text_input("Tên khoa học / Nhóm tác nhân:", value=lookup_data.get("scientific_name", ""))
                    t_cat = st.text_input("Phân loại dịch hại:", value=lookup_data.get("category", "Bệnh hại sầu riêng"))
                    t_teacher = st.text_input("Người nạp tri thức:", value=st.session_state.get("user_teacher_name", "Kỹ sư Nhà vườn"), key="input_user_teacher")
                with c_f2:
                    t_act = st.text_input("Hoạt chất bảo vệ thực vật:", value=", ".join(lookup_data.get("active_ingredients", [])))
                    t_med = st.text_input("Thuốc thương phẩm tiêu biểu:", value=", ".join(lookup_data.get("medicines", [])))
                    t_notes = st.text_input("Lưu ý chuyên môn:", value=lookup_data.get("notes", ""))
                    
                t_proto = st.text_area("Phác đồ canh tác & xử lý từng bước:", value="\n".join(lookup_data.get("protocol", [])))

                col_save1, col_save2, col_save3 = st.columns([1, 2.2, 1])
                with col_save2:
                    if st.button("💾 Ghi nhớ vào bộ nhớ máy học (Lưu tri thức & Vector 48D)", type="primary", use_container_width=True):
                        emb_list = ai_result.get("query_embedding", [])
                        if emb_list:
                            emb_arr = np.array(emb_list, dtype=np.float32)
                        else:
                            emb_arr = extract_visual_embedding(processed_bgr, leaf_mask=leaf_mask, metrics=metrics)

                        act_items = [x.strip() for x in t_act.split(",") if x.strip()]
                        med_items = [x.strip() for x in t_med.split(",") if x.strip()]
                        proto_items = [x.strip() for x in t_proto.split("\n") if x.strip()]

                        saved_item = save_learned_disease(
                            disease_name=entered_name.strip() or lookup_data.get("disease_name", "Bệnh tự học"),
                            scientific_name=t_sci.strip(),
                            category=t_cat.strip(),
                            symptoms=lookup_data.get("symptoms", ""),
                            active_ingredients=act_items,
                            medicines=med_items,
                            protocol=proto_items,
                            notes=t_notes.strip(),
                            embedding=emb_arr,
                            image_bgr=processed_bgr,
                            teacher_name=t_teacher.strip()
                        )

                        # Cập nhật kết quả chẩn đoán hiển thị ngay lập tức trên màn hình
                        learned_pred = {
                            "class_code": f"Learned_{saved_item['id']}",
                            "name_vi": saved_item["disease_name"],
                            "scientific_name": saved_item.get("scientific_name", ""),
                            "category": saved_item.get("category", "Bệnh máy đã học"),
                            "probability": 100.0,
                            "cause": f"Bệnh nhận diện từ Bộ nhớ Máy học (Ghi nhận bởi: {saved_item.get('teacher_name', 'Chuyên gia')})",
                            "symptoms": saved_item.get("symptoms", ""),
                            "favorable_conditions": "Trùng khớp hình thái học, cấu trúc đốm và phân bố màu sắc với mẫu bệnh đã lưu.",
                            "active_ingredients": saved_item.get("active_ingredients", []),
                            "medicines": saved_item.get("medicines", []),
                            "protocol": saved_item.get("protocol", []),
                            "notes": saved_item.get("notes", ""),
                            "is_learned": True,
                            "learned_id": saved_item["id"],
                            "teacher_name": saved_item.get("teacher_name", "Chuyên gia")
                        }

                        old_top = ai_result.get("top_predictions", [])
                        new_preds = [learned_pred] + [p for p in old_top if p.get("name_vi") != saved_item["disease_name"]][:2]

                        new_ai_result = {
                            "status": "SUCCESS",
                            "message": f"Nhận diện chính xác từ Bộ nhớ Máy học: {saved_item['disease_name']}",
                            "metadata": ai_result.get("metadata", {}),
                            "top1": learned_pred,
                            "top_predictions": new_preds,
                            "confidence_label": "Độ tương đồng bộ nhớ: 100.0% (Mẫu vừa được nạp & ghi nhớ)",
                            "is_ood": False,
                            "is_learned": True,
                            "query_embedding": ai_result.get("query_embedding", [])
                        }

                        st.session_state["analysis_result"] = (
                            processed_bgr, leaf_mask, spot_mask, annotated_bgr, metrics, morph_info, new_ai_result
                        )
                        st.session_state["just_learned_msg"] = f"🎉 ĐÃ GHI NHỚ THÀNH CÔNG VÀO BỘ NHỚ MÁY HỌC! Hệ thống đã nhận diện chính xác bệnh: **{saved_item['disease_name']}** (100.0%)"
                        st.session_state.pop("teach_lookup_data", None)
                        st.session_state.pop("teach_disease_name", None)
                        st.rerun()

    # =========================================================================
    # PHẦN 5: KHỐI XUẤT BÁO CÁO VÀ TẢI DỮ LIỆU
    # =========================================================================
    st.markdown("---")
    with st.expander("📥 Tải ảnh kết quả phân tích & Bảng số liệu CSV", expanded=False):
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.download_button("⬇️ Tải mask lá (PNG)", encode_image(leaf_mask), f"{image_name}_leaf_mask.png", "image/png", use_container_width=True)
        with col_d2:
            st.download_button("⬇️ Tải mask đốm (PNG)", encode_image(spot_mask), f"{image_name}_spot_mask.png", "image/png", use_container_width=True)
        with col_d3:
            st.download_button("⬇️ Tải ảnh khoanh đỏ (PNG)", encode_image(annotated_bgr), f"{image_name}_analyzed.png", "image/png", use_container_width=True)
        with col_d4:
            csv_content = (
                "image_name,disease_name,leaf_area_px,spot_area_px,severity_pct,grade\n"
                f"{image_name},{disease_name_vi if top1 else 'Unknown'},{metrics['leaf_area_px']},{metrics['spot_area_px']},{metrics['severity_percentage']},{metrics['grade']}\n"
            )
            st.download_button("📊 Tải số liệu CSV", csv_content.encode("utf-8-sig"), f"{image_name}_metrics.csv", "text/csv", use_container_width=True)




# =========================================================================
# PHẦN 6: CẨM NANG TRA CỨU TOÀN DIỆN 13 BỆNH & SÂU HẠI SẦU RIÊNG
# =========================================================================
st.markdown("---")
with st.expander("📚 Cẩm Nang Tra Cứu Toàn Diện 13 Bệnh & Sâu Hại Cây Sầu Riêng", expanded=False):
    st.markdown("""
    *Hệ tri thức bệnh học tổng hợp:* Hệ thống hỗ trợ chẩn đoán và tra cứu toàn diện **13 loại dịch hại & rối loạn sinh lý** phổ biến nhất trên cây sầu riêng tại Việt Nam (bao gồm 6 bệnh nấm cốt lõi trong bộ dữ liệu Mendeley, sâu hại chích hút, bệnh hại mạch dẫn Phytophthora và thiếu hụt dinh dưỡng).
    """)
    selected_k = st.selectbox(
        "Chọn bệnh hoặc sâu hại cần tra cứu:",
        list(EXTENDED_DISEASE_ENCYCLOPEDIA.keys()),
        format_func=lambda k: f"{EXTENDED_DISEASE_ENCYCLOPEDIA[k]['name_vi']}  [{EXTENDED_DISEASE_ENCYCLOPEDIA[k]['category']}]"
    )
    b_info = EXTENDED_DISEASE_ENCYCLOPEDIA[selected_k]
    
    st.markdown(f"### {b_info['name_vi']} *(Tên khoa học: {b_info['scientific_name']})*")
    st.caption(f"Nhóm: **{b_info['category']}**")
    st.markdown(f"**Triệu chứng lâm sàng:** {b_info['symptoms']}")
    st.markdown(f"**Nguyên nhân & điều kiện phát sinh:** {b_info['cause']}")
    st.markdown(f"**💊 Hoạt chất khuyến cáo đặc trị:** `{', '.join(b_info['active_ingredients'])}`")
    st.markdown(f"**🛒 Thuốc thương phẩm tiêu biểu:** `{', '.join(b_info['medicines'])}`")


# =========================================================================
# PHẦN 8: LỊCH SỬ CHẨN ĐOÁN LƯU TRỮ TRONG SQLITE
# =========================================================================
st.markdown("---")
with st.expander("📋 Lịch sử chẩn đoán gần đây (Cơ sở dữ liệu SQLite)", expanded=False):
    recent_records = get_recent_history(limit=15)
    if recent_records:
        df_history = pd.DataFrame(recent_records)
        display_cols = ["timestamp", "image_name", "disease_name_vi", "ai_probability", "severity_percentage", "severity_grade", "status"]
        col_rename = {
            "timestamp": "Thời gian",
            "image_name": "Tên ảnh",
            "disease_name_vi": "Chẩn đoán bệnh",
            "ai_probability": "Độ tin cậy (%)",
            "severity_percentage": "Tỷ lệ đốm (%)",
            "severity_grade": "Mức độ",
            "status": "Trạng thái"
        }
        df_display = df_history[[c for c in display_cols if c in df_history.columns]].rename(columns=col_rename)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        col_h1, col_h2 = st.columns([1, 4])
        with col_h1:
            csv_hist = df_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Xuất toàn bộ lịch sử (CSV)", csv_hist, "lich_su_chan_doan.csv", "text/csv")
        with col_h2:
            if st.button("🗑️ Xóa lịch sử"):
                clear_history()
                st.rerun()
    else:
        st.caption("Chưa có bản ghi nào. Hãy bấm 'Bắt đầu phân tích & Chẩn đoán' để ghi nhận lịch sử.")
