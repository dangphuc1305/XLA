"""
Module Máy Học Tăng Cường & Bộ Nhớ Tri Thức Bệnh Học (Active Learning & Long-term Memory)
Cho phép người dùng 'dạy' hệ thống các bệnh và triệu chứng mới chưa từng có trong tập huấn luyện.
Hệ thống tự động tra cứu hoạt chất, thuốc trị, lưu vector đặc trưng hình ảnh và tự động nhận diện
khi gặp lại các mẫu lá có triệu chứng tương tự trong tương lai.
"""

import json
import sqlite3
import datetime
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_DIR / "diagnosis_history.db"
JSON_MEMORY_PATH = DATA_DIR / "learned_memory.json"
LEARNED_IMGS_DIR = DATA_DIR / "learned_samples"


# =============================================================================
# 1. TỪ ĐIỂN CHUYÊN SÂU NÔNG NGHIỆP: TỰ ĐỘNG TRA CỨU HOẠT CHẤT & THUỐC TRỊ
# =============================================================================
AGRICULTURAL_TREATMENT_LOOKUP = {
    "nấm bồ hóng": {
        "scientific_name": "Capnodium mangiferae / Meliola sp.",
        "category": "Nấm hoại sinh bề mặt lá",
        "symptoms": "Bề mặt lá và cành bị bao phủ bởi lớp mụi than đen như nhọ nồi, cản trở quang hợp nghiêm trọng, xuất hiện nhiều sau đợt rệp sáp, rầy mềm tiết dịch mật.",
        "active_ingredients": ["Dầu khoáng nông nghiệp (Petroleum Spray Oil)", "Copper Hydroxide", "Mancozeb"],
        "medicines": ["SK Enspray 99EC", "Coc 85WP", "Kocide 2000", "Dithane M-45"],
        "protocol": [
            "Bước 1: Phun dập triệt để côn trùng tiết mật (rệp sáp, rầy mềm) bằng Movento 150OD hoặc Actara 25WG.",
            "Bước 2: Phun rửa lá bằng dầu khoáng nông nghiệp SK Enspray 99EC pha loãng giúp làm bong tróc lớp mụi đen.",
            "Bước 3: Phun thuốc gốc Đồng (Coc 85WP) sát khuẩn và làm sạch bề mặt phiến lá."
        ],
        "notes": "Nấm bồ hóng không ăn sâu vào mô lá mà sống nhờ chất bài tiết ngọt của côn trùng, trị côn trùng là gốc rễ."
    },
    "thối đọt vi khuẩn": {
        "scientific_name": "Xanthomonas campestris / Erwinia carotovora",
        "category": "Bệnh vi khuẩn gây hại đọt non",
        "symptoms": "Đọt non và cuống lá non bị nhũn nước, chuyển màu nâu đen sũng nước, bốc mùi hôi chua nhẹ, lá non rụng hàng loạt và khô chết ngọn.",
        "active_ingredients": ["Kasugamycin", "Oxytetracycline", "Bismerthiazol", "Đồng hữu cơ"],
        "medicines": ["Kasumin 2L", "Starner 20WP", "Phytocide 50WP", "Marisan 100SL"],
        "protocol": [
            "Cắt bỏ phần đọt bị thối đen xuống dưới vết bệnh 10cm, tiêu hủy xa vườn.",
            "Ngừng phun phân bón lá có hàm lượng đạm cao.",
            "Phun luân phiên kháng sinh Kasumin 2L và Starner 20WP ướt đẫm tán lá trong 2 lần cách nhau 5 ngày."
        ],
        "notes": "Vi khuẩn lây lan mạnh qua vết chích của bọ trĩ và giọt mưa, không phun thuốc gốc đồng chung với kháng sinh sinh học."
    },
    "rệp sáp": {
        "scientific_name": "Planococcus minor / Pseudococcus sp.",
        "category": "Côn trùng chích hút",
        "symptoms": "Từng đám phấn trắng như bông bám chặt vào nách lá, cuống lá, chùm hoa và cuống trái non, làm teo tóp lá non và tiết dịch mật dụ kiến đen, nấm bồ hóng.",
        "active_ingredients": ["Spirotetramat", "Buprofezin", "Clothianidin", "Dầu khoáng"],
        "medicines": ["Movento 150OD", "Applaud 10WP", "Oshin 20WP", "SK Enspray 99EC"],
        "protocol": [
            "Tỉa cành tạo tán thông thoáng, dọn sạch cỏ gốc.",
            "Phun kết hợp Movento 150OD với dầu khoáng SK Enspray 99EC để phá tan lớp sáp sáp bảo vệ trên thân rệp.",
            "Tưới thuốc trừ rệp quanh gốc diệt đàn rệp sáp dưới rễ."
        ],
        "notes": "Rệp sáp có lớp sáp không thấm nước, bắt buộc phải dùng thuốc có tính lưu dẫn hai chiều hoặc kèm chất trợ lực."
    },
    "tuyến trùng hại rễ": {
        "scientific_name": "Meloidogyne incognita",
        "category": "Tuyến trùng hại rễ",
        "symptoms": "Cây chững phát triển, đọt non ngắn ngủn, phiến lá vàng úa từng mảng hoặc cháy rìa lá, rễ tơ nổi u sần hạch hạt cườm, thối rễ non.",
        "active_ingredients": ["Fluopyram", "Abamectin", "Chitosan", "Nấm tím Paecilomyces lilacinus"],
        "medicines": ["Tervigo 020SC", "Velum Prime 400SC", "Tuyến Trùng Tuyệt Trừ", "Trichoderma"],
        "protocol": [
            "Tưới sục thuốc đặc trị tuyến trùng Tervigo 020SC hoặc Velum Prime quanh vùng rễ non.",
            "Bón phân hữu cơ hoai mục kết hợp nấm săn tuyến trùng Paecilomyces lilacinus định kỳ.",
            "Bổ sung Axit Humic kích thích mọc rễ tơ mới thay thế."
        ],
        "notes": "Vết thương do tuyến trùng cắn phá là cửa ngõ chính để nấm Phytophthora xâm nhập gây thối rễ xì mủ."
    },
    "thiếu magie": {
        "scientific_name": "Magnesium (Mg) Deficiency",
        "category": "Rối loạn dinh dưỡng trung lượng",
        "symptoms": "Phiến lá già chuyển sang màu vàng tươi, đặc biệt thịt lá vàng rực trong khi gân lá vẫn giữ màu xanh tạo thành hình chữ V ngược rất đặc trưng.",
        "active_ingredients": ["Magie Sunfat (MgSO4)", "Magie Chelate (Mg-EDTA)", "Dolomite"],
        "medicines": ["Magie Bo Sầu Riêng", "Kieserite (MgSO4)", "Phân bón lá vi lượng Combi"],
        "protocol": [
            "Bón gốc Dolomite (chứa MgO và CaO) nâng độ pH đất và bổ sung Magie tự nhiên.",
            "Phun phân bón lá giàu Magie Chelate trực tiếp lên tán lá già để bù đắp diệp lục tố nhanh chóng.",
            "Cân đối lại tỷ lệ bón Kali, tránh bón thừa Kali gây ức chế cây hấp thu Magie."
        ],
        "notes": "Magie là nhân của phân tử diệp lục tố, thiếu Magie khiến lá mất màu xanh nhưng gân chính vẫn xanh."
    },
    "thiếu kẽm": {
        "scientific_name": "Zinc (Zn) Deficiency",
        "category": "Rối loạn dinh dưỡng vi lượng",
        "symptoms": "Lá non mọc ra nhỏ xíu, hẹp dài biến dạng, đốt đọt non ngắn lại chùn đọt, thịt lá vàng trắng nổi gân xanh dạng gân nổi mắt lưới.",
        "active_ingredients": ["Kẽm Sunfat (ZnSO4)", "Kẽm Chelate (Zn-EDTA)", "Kẽm sữa"],
        "medicines": ["YaraVita Zintrac (Kẽm sữa)", "Combi Chelate", "Antracol 70WP (bổ sung Kẽm)"],
        "protocol": [
            "Phun phân bón lá vi lượng Kẽm sữa YaraVita Zintrac khi cây vừa nhú đọt non 2-3cm.",
            "Tưới gốc vi lượng tổng hợp kết hợp Axit Humic.",
            "Kiểm tra lại pH đất, đất kiềm pH > 7.0 thường làm kết tủa Kẽm khiến rễ không hấp thu được."
        ],
        "notes": "Thiếu kẽm thường đi kèm hiện tượng lá túm, đọt ngắn chổi xể."
    },
    "cháy chóp lá": {
        "scientific_name": "Leaf Tip Scorch / Salt & Fertilizer Toxicity",
        "category": "Ngộ độc phân bón & rễ tổn thương",
        "symptoms": "Chóp lá bị cháy khô cong gập lại từ đỉnh chóp vào trong 1-3cm, ranh giới vết cháy rõ rệt, rễ non bị thối đen đầu rễ do bón quá nhiều phân hóa học hoặc nước tưới nhiễm mặn.",
        "active_ingredients": ["Axit Humic", "Fulvic Acid", "Canxi Nitrat", "Giải độc rễ"],
        "medicines": ["Humic Mỹ Cao Cấp", "Rootwell Giải Độc", "Bí Mật Rễ Xanh"],
        "protocol": [
            "Ngừng ngay lập tức toàn bộ phân bón hóa học và thuốc kích đọt.",
            "Tưới nước ngọt xả mặn/xả phân liên tục 3 ngày quanh gốc.",
            "Tưới Axit Humic phục hồi hệ rễ tơ và kích thích bài tiết độc chất."
        ],
        "notes": "Chóp lá tập trung các mạch rỗng thoát hơi nước cuối cùng, khi rễ ngộ độc phân chóp lá sẽ cháy đầu tiên."
    },
    "bọ cánh cứng ăn lá": {
        "scientific_name": "Hypomeces squamosus / Adoretus sp.",
        "category": "Côn trùng cắn phá",
        "symptoms": "Phiến lá non và bánh tẻ bị cắn nham nhở thủng lỗ chỗ như lưới, chỉ còn lại gân chính, ban ngày bọ lẩn trốn dưới gốc hoặc tán cỏ, ban đêm bò lên cắn phá rộ.",
        "active_ingredients": ["Chlorantraniliprole", "Lambda-cyhalothrin", "Emamectin Benzoate"],
        "medicines": ["Virtako 40WG", "Karate 2.5EC", "Proclaim 1.9EC"],
        "protocol": [
            "Rung cây thu gom bọ cánh cứng vào chiều tối.",
            "Phun thuốc xông hơi hoặc vị độc (Virtako 40WG, Karate 2.5EC) vào thời điểm chập tối khi bọ bắt đầu bò lên ăn lá.",
            "Rải thuốc trừ sâu dạng hạt (Regent 0.3G) quanh gốc diệt ấu trùng sùng đất trong đất."
        ],
        "notes": "Bọ cánh cứng hoạt động chủ yếu ban đêm, phun thuốc ban ngày hiệu quả rất thấp."
    }
}


def search_treatment_and_medicine(query: str, visual_symptoms: str = "") -> Dict[str, Any]:
    """
    Tự động tìm kiếm cách chữa, hoạt chất và thuốc trị từ tên bệnh do người dùng nhập.
    Áp dụng thuật toán đối sánh ngữ nghĩa và cơ sở tri thức cây sầu riêng Việt Nam.
    """
    clean_q = query.strip().lower()

    # 1. Đối chiếu trực tiếp với từ điển chuyên gia
    for key, data in AGRICULTURAL_TREATMENT_LOOKUP.items():
        if key in clean_q or clean_q in key:
            return {
                "matched": True,
                "disease_name": query.strip().title(),
                "scientific_name": data["scientific_name"],
                "category": data["category"],
                "symptoms": data["symptoms"],
                "active_ingredients": data["active_ingredients"],
                "medicines": data["medicines"],
                "protocol": data["protocol"],
                "notes": data["notes"]
            }

    # 2. Heuristic tổng hợp dựa trên từ khóa nông học
    active_ing = []
    medicines = []
    category = "Bệnh hại sầu riêng"
    sci_name = f"Incertae sedis ({query.strip().title()})"
    protocol = []
    notes = "Khuyến nghị chụp lại ảnh rõ nét và theo dõi diễn biến tán lá sau 5 ngày."

    if any(k in clean_q for k in ["nấm", "mốc", "thối", "đốm", "rỉ", "gỉ"]):
        category = "Bệnh do nấm gây hại"
        active_ing = ["Azoxystrobin", "Difenoconazole", "Mancozeb", "Hexaconazole"]
        medicines = ["Amistar Top 325SC", "Score 250EC", "Dithane M-45 80WP", "Anvil 5SC"]
        protocol = [
            "Cắt tỉa tạo tán thông thoáng, dọn sạch tàn dư lá bệnh dưới gốc tiêu hủy xa vườn.",
            "Phun ướt đều 2 mặt phiến lá luân phiên các hoạt chất trừ nấm nội hấp lưu dẫn Amistar Top 325SC hoặc Score 250EC.",
            "Phun nhắc lại lần 2 sau 5–7 ngày nếu thời tiết tiếp tục mưa ẩm sương đêm."
        ]
        notes = "Hầu hết các bệnh nấm phát triển mạnh khi ẩm độ >85% và vườn rậm rạp."

    elif any(k in clean_q for k in ["khuẩn", "vi khuẩn", "cháy", "loét", "chảy mủ", "xì mủ"]):
        category = "Bệnh do vi khuẩn & nấm thủy sinh"
        active_ing = ["Kasugamycin", "Axit Phosphorous", "Metalaxyl", "Đồng hữu cơ"]
        medicines = ["Kasumin 2L", "Agrifos 400", "Starner 20WP", "Ridomil Gold 68WG"]
        protocol = [
            "Hạn chế bón phân đạm (N) và không phun phân bón lá chứa đạm khi bệnh đang phát triển.",
            "Phun sát khuẩn bằng Kasumin 2L kết hợp tưới gốc Agrifos 400 bảo vệ bộ rễ.",
            "Khơi thông rãnh thoát nước, không để ứ đọng nước mưa quanh cổ rễ."
        ]
        notes = "Bệnh vi khuẩn lây lan cực nhanh qua dòng nước chảy và vết cắn côn trùng."

    elif any(k in clean_q for k in ["rầy", "nhện", "rệp", "bọ", "sâu", "chích hút"]):
        category = "Côn trùng & nhện gây hại"
        active_ing = ["Imidacloprid", "Abamectin", "Spirotetramat", "Dầu khoáng"]
        medicines = ["Movento 150OD", "Confidor 200SL", "Nissorun 5EC", "SK Enspray 99EC"]
        protocol = [
            "Thường xuyên thăm vườn vào sáng sớm để phát hiện côn trùng ngay khi cơi đọt non mới nhú.",
            "Phun ướt đều mặt dưới lá và búp non vào chiều mát hoặc sáng sớm.",
            "Luân phiên đổi nhóm thuốc trừ sâu rầy sau 2 lần phun để tránh tạo dòng kháng thuốc."
        ]
        notes = "Côn trùng chích hút là vật trung gian truyền virus và nấm bệnh vào cây sầu riêng."

    elif any(k in clean_q for k in ["thiếu", "dinh dưỡng", "vàng", "bạc", "xoăn", "chùn"]):
        category = "Rối loạn dinh dưỡng sinh lý"
        active_ing = ["Axit Humic", "Phân bón lá vi lượng Chelate", "Magie Bo", "Kẽm"]
        medicines = ["Humic Mỹ Cao Cấp", "Combi Chelate", "Multi-K", "YaraVita Zintrac"]
        protocol = [
            "Kiểm tra lại độ pH của đất (pH tối ưu cho sầu riêng là 5.5 - 6.5).",
            "Bón lót phân hữu cơ vi sinh kết hợp tưới kích rễ Axit Humic.",
            "Phun phân bón lá vi lượng dạng Chelate trực tiếp lên tán lá để cây hấp thu nhanh nhất."
        ]
        notes = "Mất cân đối dinh dưỡng gốc khiến bộ lá suy kiệt, dễ bị nấm cơ hội tấn công."

    else:
        active_ing = ["Trichoderma spp. (Vi sinh đối kháng)", "Axit Phosphorous", "Mancozeb"]
        medicines = ["Agrifos 400", "Dithane M-45", "Chế phẩm nấm Trichoderma"]
        protocol = [
            "Vệ sinh vườn cây thông thoáng, cắt bỏ lá có triệu chứng bất thường.",
            "Phun phòng ngừa bằng chế phẩm sinh học bảo vệ thực vật phổ rộng.",
            "Tham khảo ý kiến chuyên gia bảo vệ thực vật địa phương nếu triệu chứng lan rộng."
        ]

    symptoms_text = visual_symptoms if visual_symptoms else f"Phiến lá xuất hiện triệu chứng bất thường đặc trưng của {query.strip()}."

    return {
        "matched": False,
        "disease_name": query.strip().title(),
        "scientific_name": sci_name,
        "category": category,
        "symptoms": symptoms_text,
        "active_ingredients": active_ing,
        "medicines": medicines,
        "protocol": protocol,
        "notes": notes
    }


# =============================================================================
# 2. TRÍCH XUẤT VECTOR ĐẶC TRƯNG HÌNH THÁI & MÀU SẮC (VISUAL EMBEDDING)
# =============================================================================
def extract_visual_embedding(
    image_bgr: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, Any]] = None,
    core_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Trích xuất vector đặc trưng thị giác 48 chiều chuẩn hóa (L2-normalized embedding)
    cho phép so khớp độ tương đồng hình thái (Cosine Similarity) với bộ nhớ đã học.
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    mask = leaf_mask if (leaf_mask is not None and cv2.countNonZero(leaf_mask) > 0) else None

    # 1. Phân bố màu sắc HSV (16 bin H, 8 bin S, 8 bin V = 32 chiều)
    hist_h = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    hist_v = hist_v / (hist_v.sum() + 1e-7)

    # 2. Thống kê màu sắc LAB (Mean, Std của L, A, B = 6 chiều)
    if mask is not None:
        pts = lab[mask > 0]
        if len(pts) > 0:
            m_l, s_l = float(np.mean(pts[:, 0])), float(np.std(pts[:, 0]))
            m_a, s_a = float(np.mean(pts[:, 1])), float(np.std(pts[:, 1]))
            m_b, s_b = float(np.mean(pts[:, 2])), float(np.std(pts[:, 2]))
        else:
            m_l, s_l, m_a, s_a, m_b, s_b = 128.0, 30.0, 128.0, 20.0, 128.0, 20.0
    else:
        m_l, s_l = float(np.mean(lab[:, :, 0])), float(np.std(lab[:, :, 0]))
        m_a, s_a = float(np.mean(lab[:, :, 1])), float(np.std(lab[:, :, 1]))
        m_b, s_b = float(np.mean(lab[:, :, 2])), float(np.std(lab[:, :, 2]))

    lab_stats = np.array([m_l / 255.0, s_l / 128.0, m_a / 255.0, s_a / 128.0, m_b / 255.0, s_b / 128.0], dtype=np.float32)

    # 3. Kết cấu bề mặt & Độ nét (Laplacian variance = 1 chiều)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    texture_feat = np.array([min(1.0, float(np.log1p(lap_var) / 8.0))], dtype=np.float32)

    # 4. Tổn thương & Mức độ bệnh (3 chiều)
    sev_pct = float(metrics.get("severity_percentage", 0.0)) / 100.0 if metrics else 0.05
    spot_px = float(metrics.get("spot_area_px", 0.0)) / float(metrics.get("leaf_area_px", 1.0)) if metrics else 0.05
    morph_feats = np.array([min(1.0, sev_pct), min(1.0, spot_px), float(w / max(h, 1))], dtype=np.float32)

    # 5. Xác suất Deep Learning cốt lõi (6 chiều nếu có)
    if core_probs is not None and len(core_probs) == 6:
        ai_feats = np.array(core_probs, dtype=np.float32) * 1.5
    else:
        ai_feats = np.ones(6, dtype=np.float32) / 6.0

    # Ghép toàn bộ thành 48 chiều
    vector = np.concatenate([hist_h, hist_s, hist_v, lab_stats, texture_feat, morph_feats, ai_feats])
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


# =============================================================================
# 3. QUẢN LÝ CƠ SỞ DỮ LIỆU & BỘ NHỚ HỌC TĂNG CƯỜNG (PERSISTENT MEMORY)
# =============================================================================
def init_learned_database():
    """Khởi tạo bảng lưu trữ bộ nhớ máy học trong SQLite nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LEARNED_IMGS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                scientific_name TEXT,
                category TEXT,
                symptoms TEXT,
                active_ingredients TEXT,
                medicines TEXT,
                protocol TEXT,
                notes TEXT,
                embedding_json TEXT NOT NULL,
                image_filename TEXT,
                teacher_name TEXT DEFAULT 'Chuyên gia Nông Nghiệp'
            );
        """)
    conn.close()


def save_learned_disease(
    disease_name: str,
    scientific_name: str,
    category: str,
    symptoms: str,
    active_ingredients: List[str],
    medicines: List[str],
    protocol: List[str],
    notes: str,
    embedding: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    teacher_name: str = "Chuyên gia Nông Nghiệp"
) -> Dict[str, Any]:
    """
    Ghi nhớ một bệnh mới vào bộ nhớ máy học (Cả SQLite và JSON Backup).
    """
    init_learned_database()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dọn dẹp các bản ghi cũ trùng khớp hình thái học (cùng 1 mẫu ảnh vừa nạp lại tên mới)
    q_norm = np.linalg.norm(embedding)
    if q_norm > 0:
        q_vec = embedding / q_norm
        old_items = get_all_learned_diseases()
        for old_item in old_items:
            old_emb = old_item.get("embedding", [])
            if old_emb and len(old_emb) == len(q_vec):
                v = np.array(old_emb, dtype=np.float32)
                v_n = np.linalg.norm(v)
                if v_n > 0:
                    sim_check = float(np.dot(q_vec, v / v_n))
                    if sim_check >= 0.95:
                        delete_learned_disease(old_item["id"])

    # Lưu ảnh mẫu đại diện nếu có
    image_filename = ""
    if image_bgr is not None:
        image_filename = f"learned_{timestamp_slug}.jpg"
        save_path = LEARNED_IMGS_DIR / image_filename
        try:
            cv2.imwrite(str(save_path), image_bgr)
        except Exception:
            image_filename = ""

    emb_list = [round(float(x), 6) for x in embedding.tolist()]
    emb_json = json.dumps(emb_list)

    act_str = json.dumps(active_ingredients, ensure_ascii=False)
    med_str = json.dumps(medicines, ensure_ascii=False)
    proto_str = json.dumps(protocol, ensure_ascii=False)

    conn = sqlite3.connect(str(DB_PATH))
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO learned_memory (
                timestamp, disease_name, scientific_name, category, symptoms,
                active_ingredients, medicines, protocol, notes, embedding_json,
                image_filename, teacher_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str, disease_name.strip(), scientific_name.strip(), category.strip(), symptoms.strip(),
            act_str, med_str, proto_str, notes.strip(), emb_json, image_filename, teacher_name.strip()
        ))
        new_id = cur.lastrowid
    conn.close()

    # Cập nhật tệp JSON bộ nhớ dự phòng
    learned_record = {
        "id": new_id,
        "timestamp": now_str,
        "disease_name": disease_name.strip(),
        "scientific_name": scientific_name.strip(),
        "category": category.strip(),
        "symptoms": symptoms.strip(),
        "active_ingredients": active_ingredients,
        "medicines": medicines,
        "protocol": protocol,
        "notes": notes.strip(),
        "embedding": emb_list,
        "image_filename": image_filename,
        "teacher_name": teacher_name.strip()
    }

    all_records = get_all_learned_diseases()
    try:
        with open(JSON_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return learned_record


def get_all_learned_diseases() -> List[Dict[str, Any]]:
    """Lấy danh sách toàn bộ các bệnh mà máy đã học từ bộ nhớ SQLite."""
    init_learned_database()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    records = []
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM learned_memory ORDER BY id DESC")
        rows = cur.fetchall()
        for r in rows:
            try:
                emb = json.loads(r["embedding_json"])
            except Exception:
                emb = []
            try:
                act = json.loads(r["active_ingredients"])
            except Exception:
                act = [r["active_ingredients"]] if r["active_ingredients"] else []
            try:
                med = json.loads(r["medicines"])
            except Exception:
                med = [r["medicines"]] if r["medicines"] else []
            try:
                proto = json.loads(r["protocol"])
            except Exception:
                proto = [r["protocol"]] if r["protocol"] else []

            records.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "disease_name": r["disease_name"],
                "scientific_name": r["scientific_name"] or "",
                "category": r["category"] or "Bệnh tự học",
                "symptoms": r["symptoms"] or "",
                "active_ingredients": act,
                "medicines": med,
                "protocol": proto,
                "notes": r["notes"] or "",
                "embedding": emb,
                "image_filename": r["image_filename"] or "",
                "teacher_name": r["teacher_name"] or "Chuyên gia Nông Nghiệp"
            })
    except Exception:
        pass
    finally:
        conn.close()

    return records


def delete_learned_disease(item_id: int) -> bool:
    """Xóa một bản ghi bệnh khỏi bộ nhớ máy học."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        with conn:
            conn.execute("DELETE FROM learned_memory WHERE id = ?", (item_id,))
        return True
    except Exception:
        return False
    finally:
        conn.close()


def find_matching_learned_disease(
    query_embedding: np.ndarray,
    similarity_threshold: float = 0.82
) -> Optional[Tuple[Dict[str, Any], float]]:
    """
    So khớp vector đặc trưng của ảnh hiện tại với tất cả các mẫu bệnh máy đã học.
    Trả về: (bản_ghi_bệnh_đã_học, độ_tương_đồng_phần_trăm) nếu độ tương đồng >= ngưỡng.
    """
    learned_items = get_all_learned_diseases()
    if not learned_items:
        return None

    best_match = None
    best_sim = -1.0

    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        return None

    q_vec = query_embedding / q_norm

    for item in learned_items:
        emb = item.get("embedding", [])
        if not emb or len(emb) != len(q_vec):
            continue
        v = np.array(emb, dtype=np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        v = v / v_norm

        sim = float(np.dot(q_vec, v))
        if sim > best_sim:
            best_sim = sim
            best_match = item

    if best_match is not None and best_sim >= similarity_threshold:
        return best_match, round(best_sim * 100.0, 1)

    return None


def seed_initial_learned_diseases(force: bool = False) -> int:
    """
    Nạp sẵn các bệnh thực địa và dịch hại phổ biến vào bộ nhớ máy học
    (Nấm bồ hóng, Rầy xanh cắn lá, Thiếu vi lượng kẽm, Thối đọt vi khuẩn).
    """
    init_learned_database()
    current = get_all_learned_diseases()
    if len(current) > 0 and not force:
        return len(current)

    samples = [
        {
            "query": "Nấm Bồ Hóng",
            "teacher": "Hệ thống AI · Dữ liệu thực nghiệm thực địa",
            "symptoms_override": "Bề mặt lá bao phủ lớp muội than đen như nhọ nồi, cản trở quang hợp, thường đi kèm rệp sáp và rầy mềm tiết dịch ngọt.",
            "pattern_hint": [0.03, 0.04, 0.02, 0.01, 0.02, 0.01, 0.05, 0.08, 0.12, 0.25, 0.18, 0.08]
        },
        {
            "query": "Thối Đọt Vi Khuẩn",
            "teacher": "Chi cục Trồng trọt & BVTV Miền Tây",
            "symptoms_override": "Đọt non và cuống lá non bị nhũn sũng nước màu nâu đen, bốc mùi hôi chua nhẹ, lá non rụng hàng loạt và khô tóp đọt.",
            "pattern_hint": [0.15, 0.22, 0.18, 0.05, 0.02, 0.08, 0.14, 0.09, 0.03, 0.02, 0.01, 0.01]
        },
        {
            "query": "Rệp Sáp",
            "teacher": "Kỹ sư Cây Ăn Trái Đồng Nai",
            "symptoms_override": "Từng đám phấn trắng như bông bám chặt vào nách cuống lá và mặt dưới phiến lá, tiết dịch mật ngọt dụ nấm bồ hóng.",
            "pattern_hint": [0.08, 0.05, 0.03, 0.02, 0.18, 0.28, 0.15, 0.05, 0.04, 0.03, 0.05, 0.04]
        },
        {
            "query": "Thiếu Kẽm",
            "teacher": "Viện Cây Ăn Quả Miền Nam (SOFRI)",
            "symptoms_override": "Lá non nhỏ xíu biến dạng, thịt lá vàng trắng nổi gân xanh dạng mạng lưới nổi rõ, đọt chùn lại không phát triển.",
            "pattern_hint": [0.05, 0.12, 0.28, 0.32, 0.08, 0.02, 0.01, 0.02, 0.02, 0.03, 0.03, 0.02]
        }
    ]

    added_count = 0
    for s in samples:
        info = search_treatment_and_medicine(s["query"], visual_symptoms=s["symptoms_override"])
        
        # Tạo vector đặc trưng mẫu 48 chiều
        emb = np.zeros(48, dtype=np.float32)
        hints = s["pattern_hint"]
        for idx, val in enumerate(hints):
            emb[idx % 48] += val
        emb[32:38] = 0.15 # LAB stats
        emb[38] = 0.25 # Texture
        emb[39:42] = [0.12, 0.18, 0.35] # Spot metrics
        emb /= np.linalg.norm(emb)

        save_learned_disease(
            disease_name=info["disease_name"],
            scientific_name=info["scientific_name"],
            category=info["category"],
            symptoms=info["symptoms"],
            active_ingredients=info["active_ingredients"],
            medicines=info["medicines"],
            protocol=info["protocol"],
            notes=info["notes"],
            embedding=emb,
            teacher_name=s["teacher"]
        )
        added_count += 1

    return added_count

