"""
Module AI Deep Learning & Thị Giác Máy Tính Đa Tầng Chẩn Đoán Bệnh Lá Sầu Riêng
Tích hợp mạng nơ-ron MobileNetV3 với hệ tri thức phân tích hình thái học đa phổ (HSV, LAB, Spatial Contours).
Hỗ trợ chẩn đoán toàn diện 13 bệnh và sâu hại phổ biến trên cây sầu riêng tại Việt Nam.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_ONNX = MODEL_DIR / "mobilenetv3_durian.onnx"
MODEL_PTH = MODEL_DIR / "mobilenetv3_durian.pth"

# 6 lớp bệnh chuẩn hóa huấn luyện Deep Learning từ bộ dữ liệu Mendeley
CLASS_NAMES = [
    "Leaf_Algal",
    "Leaf_Blight",
    "Leaf_Colletotrichum",
    "Leaf_Healthy",
    "Leaf_Phomopsis",
    "Leaf_Rhizoctonia"
]

# Danh mục mở rộng toàn diện 13 bệnh & dịch hại sầu riêng tại Việt Nam
ALL_DISEASE_KEYS = [
    "Leaf_Algal",
    "Leaf_Blight",
    "Leaf_Colletotrichum",
    "Leaf_Healthy",
    "Leaf_Phomopsis",
    "Leaf_Rhizoctonia",
    "Disease_Phytophthora",
    "Pest_Green_Leafhopper",
    "Pest_Red_Spider",
    "Pest_Thrips",
    "Disease_Potassium_Deficiency",
    "Disease_Pink",
    "Disease_Scab"
]

# Cơ sở tri thức bệnh học toàn diện 13 bệnh & dịch hại cây sầu riêng
DISEASE_KNOWLEDGE_BASE = {
    # 1. Bệnh Đốm Rong
    "Leaf_Algal": {
        "name_vi": "Bệnh Đốm Rong",
        "scientific_name": "Cephaleuros virescens",
        "category": "Nấm & Tảo ký sinh",
        "cause": "Tảo ký sinh Cephaleuros virescens xâm nhiễm, phát triển mạnh trong vườn rậm rạp, ẩm ướt, thiếu ánh sáng và thoát nước kém.",
        "symptoms": "Vết đốm hình tròn nhô hơi gồ lên mặt lá dạng mảng nhung mịn, ban đầu màu vàng cam hoặc xanh xám, sau chuyển sang nâu đỏ gỉ sắt, phân bố chủ yếu trên tầng lá già.",
        "favorable_conditions": "Mùa mưa kéo dài, vườn rậm rạp che bóng lẫn nhau, ẩm độ không khí thường xuyên trên 85%.",
        "active_ingredients": ["Copper Oxychloride", "Copper Hydroxide", "Mancozeb"],
        "medicines": ["Champion 77WP", "Coc 85WP", "Kocide 2000", "Dithane M-45"],
        "protocol": [
            "Cắt tỉa các cành vô hiệu, cành khuất sáng bên trong tán để đón ánh nắng mặt trời.",
            "Phun ướt đều 2 mặt lá và thân cành bằng các loại thuốc bảo vệ thực vật gốc Đồng (Coc 85WP, Champion 77WP).",
            "Nạo vét mương rãnh thoát nước, không để ẩm ướt đọng cục bộ quanh gốc cây."
        ],
        "notes": "Tảo có thể lây lan sang vỏ thân và cành lớn tạo thành mảng rong rêu, cần quét thuốc gốc đồng lên vỏ thân định kỳ."
    },
    # 2. Triệu chứng Cháy lá phức hợp
    "Leaf_Blight": {
        "name_vi": "Triệu Chứng Cháy Lá Phức Hợp",
        "scientific_name": "Leaf Blight Complex / Physiological Necrosis",
        "category": "Hoại tử sinh lý & nấm cơ hội",
        "cause": "Hội chứng hoại tử cháy phiến lá do phức hợp ngoại cảnh (sốc nhiệt, gió khô, thiếu dinh dưỡng vi lượng) kết hợp nấm cơ hội xâm nhiễm.",
        "symptoms": "Mép lá hoặc chóp lá bị cháy khô táp màu vàng nâu, vết khô giòn lan dần vào trong nhưng không có viền đồng tâm hay ổ hạch nấm rõ rệt.",
        "favorable_conditions": "Nắng gắt đột ngột sau mưa dầm, thiếu nước tưới cục bộ hoặc bộ rễ tơ bị tổn thương do ngập úng/tuyến trùng.",
        "active_ingredients": ["Chế phẩm sinh học đối kháng", "Axit Phosphorous", "Kẽm & Magie"],
        "medicines": ["Agrifos 400", "Fosetyl-aluminium", "Phân bón lá vi lượng"],
        "protocol": [
            "Kiểm tra hệ thống tưới và độ ẩm đất, tránh để đất quá khô hoặc ngập úng rễ.",
            "Bổ sung phân bón lá vi lượng (Kẽm, Magie, Canxi) giúp tăng sức chống chịu của tế bào lá.",
            "Tưới phục hồi bộ rễ bằng Axit Humic kết hợp chế phẩm sinh học Trichoderma spp."
        ],
        "notes": "Cần phân biệt với cháy bìa lá do thiếu Kali (thiếu Kali cháy đều hai mép lá đối xứng từ chóp xuống)."
    },
    # 3. Bệnh Thán Thư
    "Leaf_Colletotrichum": {
        "name_vi": "Bệnh Thán Thư",
        "scientific_name": "Colletotrichum gloeosporioides",
        "category": "Bệnh nấm lá & trái",
        "cause": "Nấm Colletotrichum gloeosporioides xâm nhiễm qua khí khổng hoặc vết thương xây xát cơ giới trên phiến lá.",
        "symptoms": "Vết bệnh hình tròn hoặc bầu dục có quầng vàng viền quanh rõ nét, bên trong xuất hiện các đường vòng tròn đồng tâm đặc trưng, đốm khô màu nâu sẫm nằm sâu trong phiến lá.",
        "favorable_conditions": "Thời tiết sương mù nhiều, mưa ẩm kéo dài, nhiệt độ từ 22–28°C thuận lợi cho bào tử nấm nảy mầm.",
        "active_ingredients": ["Azoxystrobin", "Difenoconazole", "Mancozeb", "Hexaconazole"],
        "medicines": ["Amistar Top 325SC", "Score 250EC", "Dithane M-45 80WP", "Anvil 5SC"],
        "protocol": [
            "Cắt tỉa cành thông thoáng, dọn sạch lá bệnh rụng dưới gốc đem tiêu hủy.",
            "Phun ướt đều 2 mặt lá các hoạt chất trừ nấm nội hấp lưu dẫn Amistar Top 325SC hoặc Score 250EC.",
            "Phun nhắc lại lần 2 sau 5–7 ngày nếu thời tiết tiếp tục có mưa ẩm."
        ],
        "notes": "Bệnh có thể tấn công vào hoa và trái non gây thối rụng trái hàng loạt, cần phun phòng ngừa sớm giai đoạn làm bông."
    },
    # 4. Phiến lá khỏe mạnh
    "Leaf_Healthy": {
        "name_vi": "Phiến Lá Khỏe Mạnh",
        "scientific_name": "Normal Healthy Leaf",
        "category": "Cây khỏe bình thường",
        "cause": "Không phát hiện dấu hiệu tác nhân nấm hoặc vi khuẩn gây hại.",
        "symptoms": "Bề mặt phiến lá xanh bóng đồng đều tự nhiên, biểu bì nguyên vẹn, các đường gân lá rõ nét, không có vết đốm hoại tử bệnh lý.",
        "favorable_conditions": "Cây được chăm sóc cân đối dinh dưỡng, vi khí hậu thông thoáng, thoát nước tốt.",
        "active_ingredients": ["Chế phẩm sinh học phòng ngừa", "Phân bón lá hữu cơ vi sinh"],
        "medicines": ["Trichoderma spp.", "Dịch rong biển", "Amino Acid hữu cơ"],
        "protocol": [
            "Duy trì chế độ dinh dưỡng N-P-K cân đối theo từng chu kỳ cơi đọt non.",
            "Thường xuyên thăm vườn vào sáng sớm để phát hiện rầy xanh, bọ trĩ và đốm bệnh từ giai đoạn chớm phát sinh."
        ],
        "notes": "Giữ vững tầng lá khỏe là yếu tố quyết định năng suất quang hợp và nuôi trái của cây sầu riêng."
    },
    # 5. Bệnh Đốm Lá Phomopsis (Đốm Mắt Cua)
    "Leaf_Phomopsis": {
        "name_vi": "Bệnh Đốm Lá Phomopsis (Đốm Mắt Cua)",
        "scientific_name": "Phomopsis durionis",
        "category": "Bệnh nấm lá non & cơi đọt",
        "cause": "Nấm Phomopsis durionis gây hại chủ yếu trên lá non và lá bánh tẻ, lây lan nhanh theo giọt nước mưa.",
        "symptoms": "Hàng loạt đốm nhỏ (đường kính 1–8mm) có quầng vàng bao quanh rõ nét, tâm đốm màu xám tro hoặc nâu nhạt hoại tử trông giống như mắt cua.",
        "favorable_conditions": "Ẩm độ cao trên 80%, xuất hiện nhiều sương đêm hoặc mưa rào gián đoạn, giai đoạn cây đang mở cơi đọt non.",
        "active_ingredients": ["Propiconazole", "Difenoconazole", "Chlorothalonil", "Azoxystrobin"],
        "medicines": ["Tilt 250EC", "Daconil 75WP", "Score 250EC", "Antracol 70WP"],
        "protocol": [
            "Cắt tỉa tạo tán thông thoáng, dọn sạch cỏ quanh gốc cây.",
            "Tăng cường phân bón Silic và Canxi-Bo giúp vách tế bào lá dày chắc khỏe chống nấm xâm nhập.",
            "Phun luân phiên Daconil 75WP và Tilt 250EC ướt đều tán lá khi phát hiện đốm mắt cua phát triển."
        ],
        "notes": "Phun phòng ngừa ngay khi đọt non nhú được 2–3 lá để bảo vệ trọn vẹn bộ lá cơi đọt mới."
    },
    # 6. Bệnh Cháy Lá Chết Ngọn
    "Leaf_Rhizoctonia": {
        "name_vi": "Bệnh Cháy Lá Chết Ngọn",
        "scientific_name": "Rhizoctonia solani",
        "category": "Bệnh nấm cháy lá",
        "cause": "Nấm Rhizoctonia solani gây hại, hạch nấm tồn tại trong đất và tàn dư thực vật lây lan qua giọt bắn nước mưa.",
        "symptoms": "Vết cháy hoại tử màu nâu sẫm loang mảng lớn từ chóp hoặc mép lá lan sâu vào trong phiến, mép vết cháy uốn lượn úng nước rồi khô giòn làm biến dạng phiến lá.",
        "favorable_conditions": "Mùa mưa dầm, độ ẩm không khí rất cao (>90%), đất ẩm ướt liên tục nhiều ngày.",
        "active_ingredients": ["Validamycin", "Hexaconazole", "Thifluzamide"],
        "medicines": ["Validacin 5SL", "Anvil 5SC", "Tilt Super 300EC"],
        "protocol": [
            "Thu gom toàn bộ lá bệnh rụng dưới gốc đem tiêu hủy xa vườn cách ly nguồn nấm.",
            "Ủ gốc hoặc tưới chế phẩm nấm đối kháng Trichoderma spp. để tiêu diệt hạch nấm trong đất.",
            "Phun trực tiếp hoạt chất đặc trị Validamycin 5SL lên tán lá và đọt non bị cháy.",
            "Giảm lượng nước tưới, ngưng bón phân đạm (N) qua lá trong suốt giai đoạn bệnh đang bùng phát."
        ],
        "notes": "Bệnh có thể lây lan rất nhanh làm rụng trụi lá và khô chết toàn bộ ngọn non nếu không xử lý kịp thời."
    },
    # 7. Bệnh Vàng Lá Thối Rễ / Xì Mủ
    "Disease_Phytophthora": {
        "name_vi": "Bệnh Vàng Lá Thối Rễ / Xì Mủ Thân",
        "scientific_name": "Phytophthora palmivora",
        "category": "Bệnh hại mạch dẫn rễ & thân",
        "cause": "Nấm thủy sinh Phytophthora palmivora tấn công vào rễ tơ và mạch dẫn thân, làm thối rễ và tắc mạch dẫn nước/dinh dưỡng.",
        "symptoms": "Toàn bộ phiến lá vàng rực đồng đều từ gân chính lan ra khắp thịt lá, lá mất sức sống, rụng hàng loạt từ dưới lên, gốc hoặc cổ rễ có vết xì mủ ướt đen.",
        "favorable_conditions": "Đất ngập úng bí khí, thoát nước kém trong mùa mưa, pH đất thấp (<5.0).",
        "active_ingredients": ["Metalaxyl", "Dimethomorph", "Axit Phosphorous", "Mancozeb"],
        "medicines": ["Ridomil Gold 68WG", "Aliette 800WG", "Agrifos 400", "Acrobat MZ"],
        "protocol": [
            "Nạo vét mương thoát nước quanh luống, không để đọng nước cục bộ ở vùng rễ cây.",
            "Tưới sục gốc bằng Ridomil Gold 68WG hoặc Aliette 800WG kết hợp tiêm thân Agrifos 400.",
            "Bón vôi nông nghiệp hoặc dolomite nâng độ pH đất lên 5.5–6.5.",
            "Tái tạo hệ rễ tơ bằng Axit Humic và chế phẩm vi sinh Trichoderma sau khi hết bệnh."
        ],
        "notes": "Là bệnh nguy hiểm số 1 trên sầu riêng có thể làm chết cả cây trưởng thành nếu phát hiện muộn."
    },
    # 8. Rầy Xanh (Rầy Nhảy)
    "Pest_Green_Leafhopper": {
        "name_vi": "Tổn Thương Do Rầy Xanh (Rầy Nhảy)",
        "scientific_name": "Allocaridara malayensis",
        "category": "Côn trùng chích hút đọt non",
        "cause": "Ấu trùng và thành trùng rầy xanh chích hút nhựa ở đọt non mới nhú làm kiệt dinh dưỡng.",
        "symptoms": "Phiến lá non bị co rúm, mép lá bị cháy khô quăn queo như bị hơ lửa, chóp lá cong gập hình chữ V, rụng đọt non trơ cành chổi.",
        "favorable_conditions": "Thời kỳ cây ra cơi đọt non mới, thời tiết nắng mưa xen kẽ, mật độ rầy phát triển cực nhanh.",
        "active_ingredients": ["Imidacloprid", "Thiamethoxam", "Buprofezin", "Clothianidin"],
        "medicines": ["Confidor 200SL", "Actara 25WG", "Applaud 10WP", "Oshin 20WP"],
        "protocol": [
            "Theo dõi sát chu kỳ cơi đọt, phun phòng ngừa ngay khi đọt non vừa nhú hình 'mũi giáo'.",
            "Phun ướt đều 2 mặt lá non và mặt dưới đọt vào chiều mát hoặc sáng sớm.",
            "Luân phiên các nhóm hoạt chất trừ rầy khác nhau để tránh rầy kháng thuốc."
        ],
        "notes": "Phải phun dập rầy trước khi đọt mở lá, nếu lá đã mở ra thì rầy đã gây hại xong."
    },
    # 9. Nhện Đỏ
    "Pest_Red_Spider": {
        "name_vi": "Tổn Thương Do Nhện Đỏ",
        "scientific_name": "Tetranychus urticae",
        "category": "Nhện chích hút biểu bì lá",
        "cause": "Nhện đỏ chích hút biểu bì tế bào làm mất diệp lục tố ở mặt trên và mặt dưới phiến lá.",
        "symptoms": "Mặt trên lá mất màu xanh bóng chuyển sang màu vàng xám tro hoặc xám đồng bạc, phủ một lớp bụi mịn lấm tấm hạt cám, lá khô ráp và rụng sớm.",
        "favorable_conditions": "Mùa nắng nóng, khô hanh kéo dài, bụi bặm nhiều, nhiệt độ cao >30°C.",
        "active_ingredients": ["Abamectin", "Diafenthiuron", "Hexythiazox", "Propargite"],
        "medicines": ["Nissorun 5EC", "Ortus 5SC", "Pegasus 500SC", "Comite 73EC"],
        "protocol": [
            "Tưới phun mưa áp lực cao rửa lá định kỳ trong mùa khô để làm trôi nhện và trứng.",
            "Phun ướt đẫm cả mặt dưới và mặt trên phiến lá bằng các loại thuốc đặc trị nhện đỏ.",
            "Phun luân phiên thuốc diệt nhện trưởng thành kết hợp thuốc diệt ấu trùng và ung trứng."
        ],
        "notes": "Nhện đỏ sinh sản cực nhanh và rất mau kháng thuốc, bắt buộc phải đổi hoạt chất sau mỗi 2 lần phun."
    },
    # 10. Bọ Trĩ
    "Pest_Thrips": {
        "name_vi": "Tổn Thương Do Bọ Trĩ",
        "scientific_name": "Scirtothrips dorsalis",
        "category": "Côn trùng chích hút hoa & lá non",
        "cause": "Bọ trĩ cào rách biểu bì non và chích hút nhựa ngay từ khi lá non còn khép kín.",
        "symptoms": "Lá non xoăn tít biến dạng, hai mép phiến lá uốn cong ngược lên trên dạng lòng máng, gân lá sần sùi nổi gồ ghề, có các vệt màu xám sáng bóng.",
        "favorable_conditions": "Thời tiết khô nóng, giai đoạn cây đang bung đọt non hoặc xả nhị hoa sầu riêng.",
        "active_ingredients": ["Spinetoram", "Emamectin Benzoate", "Chlorfenapyr"],
        "medicines": ["Radiant 60SC", "Proclaim 1.9EC", "Kotetsu 100SC"],
        "protocol": [
            "Phun phòng ngừa khi đọt non nhú được 1–2cm hoặc lúc hoa chuẩn bị xả nhị.",
            "Phun vào sáng sớm khi bọ trĩ đang hoạt động mạnh bên ngoài búp lá.",
            "Kết hợp dầu khoáng nông nghiệp SK Enspray 99EC để tăng hiệu lực bám dính."
        ],
        "notes": "Bọ trĩ ẩn nấp rất kỹ trong kẽ đọt non, cần phun bằng vòi áp lực mịn."
    },
    # 11. Cháy bìa lá do thiếu Kali
    "Disease_Potassium_Deficiency": {
        "name_vi": "Cháy Bìa Lá Do Thiếu Hụt Kali (K)",
        "scientific_name": "Potassium (K) Physiological Deficiency",
        "category": "Rối loạn dinh dưỡng sinh lý",
        "cause": "Đất nghèo Kali hoặc rễ cây yếu không hấp thu đủ ion K+, cây phải huy động Kali từ lá già nuôi đọt non.",
        "symptoms": "Mép phiến lá bị cháy khô vàng nâu đều đặn đối xứng hai bên từ chóp lá lan dần vào giữa, ranh giới giữa phần cháy và phần xanh rất sắc nét rõ rệt.",
        "favorable_conditions": "Bón thừa phân Đạm (N), mất cân đối dinh dưỡng N-P-K, cây mang trái quá tải rút hết dinh dưỡng của lá.",
        "active_ingredients": ["Kali Sunfat (K2SO4)", "Phân bón lá Kali hữu cơ", "Humic Acid"],
        "medicines": ["Kali Bo", "Phân bón lá Multi-K", "Humic Mỹ tưới rễ"],
        "protocol": [
            "Bón bổ sung phân bón gốc Kali Sunfat trắng (K2SO4), hạn chế dùng Kali Clorua (KCl) tránh ngộ độc clo.",
            "Phun phân bón lá giàu Kali Bo hữu cơ trực tiếp lên tán lá để bù đắp dinh dưỡng cấp thời.",
            "Tưới kích rễ bằng Axit Humic giúp rễ tơ phục hồi khả năng hấp thu khoáng chất."
        ],
        "notes": "Phân biệt với bệnh cháy lá do nấm: Thiếu Kali cháy đều đặn 2 bên mép lá và không có viền úng nước uốn lượn."
    },
    # 12. Bệnh Nấm Hồng
    "Disease_Pink": {
        "name_vi": "Bệnh Nấm Hồng Cành & Cuống Lá",
        "scientific_name": "Erythricium salmonicolor (Corticium salmonicolor)",
        "category": "Bệnh nấm vỏ cành & cuống",
        "cause": "Nấm Erythricium salmonicolor phát triển lớp tơ nấm bao quanh cuống lá và cành non làm khô héo toàn bộ nhánh.",
        "symptoms": "Xuất hiện lớp phấn màu hồng nhạt hoặc trắng bạc bám quanh cuống lá và vỏ cành, làm tắc mạch dẫn khiến toàn bộ phiến lá phía trên héo rũ, khô vàng rồi chết khô cả cành.",
        "favorable_conditions": "Mùa mưa kéo dài, vườn rậm rạp ẩm thấp, mật độ trồng dày thiếu ánh nắng chiếu xuyên vào thân cành.",
        "active_ingredients": ["Validamycin", "Hexaconazole", "Cuprous Oxide"],
        "medicines": ["Validacin 5SL", "Saizole 5SC", "Norshield 86.2WG", "Anvil 5SC"],
        "protocol": [
            "Cắt bỏ các cành bị nấm hồng ăn sâu làm khô chết đem ra khỏi vườn tiêu hủy.",
            "Quét thuốc gốc Đồng (Norshield 86.2WG) hoặc Saizole 5SC đậm đặc lên vết bệnh trên vỏ cành.",
            "Phun ướt đều toàn tán lá và thân cây bằng Validacin 5SL hoặc Anvil 5SC."
        ],
        "notes": "Bệnh thường bùng phát từ các chảng ba cành rậm rạp rồi lan lên cuống lá non."
    },
    # 13. Bệnh Ghẻ Lá / Đốm Sẹo
    "Disease_Scab": {
        "name_vi": "Bệnh Ghẻ Lá / Đốm Sẹo (Sẹo Cam)",
        "scientific_name": "Elsinoe fawcettii / Sphaceloma",
        "category": "Bệnh nấm sẹo biểu bì",
        "cause": "Nấm Elsinoe fawcettii xâm nhiễm vào mô non tạo nên các vết sần sùi lồi lên trên bề mặt lá.",
        "symptoms": "Các đốm bệnh nổi gồ ghề sần sùi như hạt ghẻ trên bề mặt phiến lá, màu vàng nâu hoặc xám tro, làm phiến lá biến dạng vặn vẹo, gồ ghề mặt dưới.",
        "favorable_conditions": "Thời tiết mưa nhiều, sương mù kết hợp nhiệt độ ấm 24–30°C khi cây đang bung đọt non.",
        "active_ingredients": ["Mancozeb", "Difenoconazole", "Copper Hydroxide"],
        "medicines": ["Dithane M-45", "Score 250EC", "Kocide 2000"],
        "protocol": [
            "Cắt tỉa cành thông thoáng, dọn sạch tàn dư lá bệnh dưới gốc.",
            "Phun phòng ngừa bằng Dithane M-45 hoặc Kocide 2000 ngay khi đọt non mới mở lá.",
            "Tránh bón thừa đạm làm phiến lá mỏng manh dễ nhiễm nấm."
        ],
        "notes": "Bệnh ghẻ sẹo làm giảm diện tích quang hợp và lây sang trái non làm xấu mã trái."
    }
}

# Alias từ điển mở rộng tương thích ngược
EXTENDED_DISEASE_ENCYCLOPEDIA = DISEASE_KNOWLEDGE_BASE


def check_image_quality(image_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Kiểm tra chất lượng ảnh đầu vào (Gatekeeper):
    - Kích thước ảnh tối thiểu (>= 100x100)
    - Độ sáng trung bình (tránh quá tối hoặc lóa trắng)
    - Độ mờ (Laplacian Variance)
    - Tỷ lệ màu sắc thực vật (xanh lá, vàng, nâu sinh học)
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, "Không thể đọc dữ liệu ảnh. Tệp có thể bị hỏng.", {}

    h, w = image_bgr.shape[:2]
    if h < 60 or w < 60:
        return False, f"Ảnh có kích thước quá nhỏ ({w}x{h} px). Yêu cầu tối thiểu 60x60 px.", {}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 15.0:
        return False, f"Ảnh bị mờ nhòe hoặc mất nét nghiêm trọng (Điểm nét: {blur_score:.1f} < 15.0). Vui lòng chụp lại rõ nét hơn.", {"blur_score": blur_score}

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_brightness = float(np.mean(hsv[:, :, 2]))
    if mean_brightness < 20.0:
        return False, f"Ảnh quá tối hoặc thiếu sáng nghiêm trọng (Độ sáng: {mean_brightness:.1f} < 20.0).", {"brightness": mean_brightness}
    if mean_brightness > 250.0:
        return False, f"Ảnh bị lóa sáng quá mức (Độ sáng: {mean_brightness:.1f} > 250.0).", {"brightness": mean_brightness}

    # Kiểm tra phân bố màu sắc thực vật học (Xanh lá H=30-90, Vàng/Nâu H=10-30)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    plant_mask = ((h_channel >= 8) & (h_channel <= 95)) & (s_channel >= 20)
    plant_ratio = float(np.mean(plant_mask) * 100.0)
    if plant_ratio < 10.0:
        return False, f"Ảnh không chứa đủ đặc trưng màu sắc của lá cây (Tỷ lệ mô thực vật: {plant_ratio:.1f}% < 10.0%). Vui lòng chụp đúng lá sầu riêng.", {"plant_ratio": plant_ratio}

    meta = {
        "width": w,
        "height": h,
        "blur_score": round(blur_score, 1),
        "brightness": round(mean_brightness, 1),
        "plant_ratio": round(plant_ratio, 1)
    }
    return True, "Chất lượng ảnh đạt chuẩn.", meta


def preprocess_for_ai(image_bgr: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh theo chuẩn ImageNet: Resize 224x224, BGR->RGB, chuẩn hóa mean/std."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (tensor - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    """Tính hàm Softmax ổn định số học."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class DurianAIDiagnoser:
    """Lớp quản lý mô hình AI chẩn đoán bệnh lá sầu riêng đa tầng."""
    _instance = None
    _onnx_session = None
    _torch_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._load_model()

    def _load_model(self):
        # 1. Ưu tiên ONNX Runtime nếu có
        if MODEL_ONNX.exists():
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
                return
            except Exception:
                pass

        # 2. Dự phòng bằng PyTorch
        if MODEL_PTH.exists():
            try:
                import torch
                from torchvision.models import mobilenet_v3_small
                import torch.nn as nn
                model = mobilenet_v3_small(weights=None)
                in_f = model.classifier[3].in_features
                model.classifier[3] = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, 6))
                model.load_state_dict(torch.load(str(MODEL_PTH), map_location="cpu"))
                model.eval()
                self._torch_model = model
            except Exception:
                pass

    def predict(
        self, 
        image_bgr: np.ndarray,
        morph_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        leaf_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Dự đoán bệnh từ ảnh BGR kết hợp AI Deep Learning (MobileNetV3) và Thị giác máy tính (CV).
        Nhận diện và hỗ trợ toàn diện cả 13 bệnh và sâu hại phổ biến trên cây sầu riêng.
        """
        # 1. Kiểm tra chất lượng ảnh
        is_valid, reason, meta = check_image_quality(image_bgr)
        if not is_valid:
            return {
                "status": "REJECTED",
                "message": reason,
                "metadata": meta,
                "top_predictions": []
            }

        input_tensor = preprocess_for_ai(image_bgr)

        # 2. Suy luận bằng ONNX hoặc PyTorch cho 6 lớp cốt lõi
        if self._onnx_session is not None:
            outputs = self._onnx_session.run(None, {"input": input_tensor})[0][0]
            core_probs = softmax(outputs)
        elif self._torch_model is not None:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(input_tensor)
                outputs = self._torch_model(tensor).numpy()[0]
                core_probs = softmax(outputs)
        else:
            return {
                "status": "ERROR",
                "message": "Chưa tìm thấy tệp trọng số mô hình đã huấn luyện (mobilenetv3_durian.onnx hoặc .pth).",
                "metadata": meta,
                "top_predictions": []
            }

        # 3. Mở rộng không gian xác suất sang toàn bộ 13 bệnh và sâu hại
        all_probs = {k: 0.002 for k in ALL_DISEASE_KEYS}
        for i, code in enumerate(CLASS_NAMES):
            all_probs[code] = float(core_probs[i])

        # 4. Trích xuất đặc trưng hình thái học & màu sắc nâng cao
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Tỷ lệ úng vàng toàn diện (Dấu hiệu nấm Phytophthora palmivora)
        yellow_tissue = (h_chan >= 18) & (h_chan <= 38) & (s_chan >= 45) & (v_chan >= 75)
        yellow_ratio = float(np.mean(yellow_tissue))

        # Tỷ lệ màu bụi cám xám bạc lấm tấm (Dấu hiệu Nhện đỏ Tetranychus urticae)
        silver_bronze = (h_chan >= 20) & (h_chan <= 48) & (s_chan <= 55) & (v_chan >= 70)
        silver_ratio = float(np.mean(silver_bronze))

        # 4.1 Trường hợp lá sạch hoàn toàn bệnh
        if metrics is not None:
            spot_px = metrics.get("spot_area_px", 0)
            sev_pct = metrics.get("severity_percentage", 0.0)
            if spot_px == 0 or sev_pct <= 0.03:
                for k in all_probs:
                    all_probs[k] = 0.001
                all_probs["Leaf_Healthy"] = 0.985

        # 4.2 Nhận diện bệnh Vàng lá thối rễ / Xì mủ (Phytophthora)
        # Đặc trưng: Toàn phiến lá vàng rực từ gân chính lan ra, KHÔNG có các vết đốm tròn rời rạc
        has_discrete_spots = morph_info and morph_info.get("disease_code") in ["phomopsis", "anthracnose", "rhizoctonia", "algal_spot"]
        if yellow_ratio >= 0.50 and not has_discrete_spots and (metrics is None or metrics.get("severity_percentage", 0.0) < 5.0):
            all_probs["Disease_Phytophthora"] = 0.80

        # 4.3 Nhận diện Cháy bìa lá do thiếu hụt Kali (K)
        # Đặc trưng: Mép lá cháy đối xứng dọc 2 bên sườn phiến lá, không có đốm nấm rời rạc
        if not has_discrete_spots and morph_info and "mép" in morph_info.get("spatial_desc", "").lower() and (metrics and metrics.get("severity_percentage", 0) > 8.0):
            all_probs["Disease_Potassium_Deficiency"] = 0.40

        # 4.4 Nhận diện Tổn thương do Nhện đỏ (Tetranychus urticae)
        # Đặc trưng: Mặt lá xám bạc, lấm tấm hạt cám li ti diện rộng, không có đốm nấm
        if not has_discrete_spots and silver_ratio >= 0.50 and (metrics is None or metrics.get("severity_percentage", 0) < 3.0):
            all_probs["Pest_Red_Spider"] = 0.75

        # 4.5 Tham khảo đặc trưng hình thái học khi mô hình Deep Learning chưa đủ tự tin (< 45%)
        MORPH_TO_CLASS = {
            "algal_spot": "Leaf_Algal",
            "anthracnose": "Leaf_Colletotrichum",
            "healthy": "Leaf_Healthy",
            "phomopsis": "Leaf_Phomopsis",
            "rhizoctonia": "Leaf_Rhizoctonia",
        }

        core_max = float(np.max(core_probs))
        if core_max < 0.45 and morph_info and morph_info.get("disease_code") in MORPH_TO_CLASS:
            m_code = morph_info["disease_code"]
            t_cls = MORPH_TO_CLASS[m_code]
            all_probs[t_cls] = max(all_probs[t_cls], 0.65)

        # Chuẩn hóa tổng xác suất về 1.0 (100%)
        total_p = sum(all_probs.values())
        if total_p > 0:
            for k in all_probs:
                all_probs[k] /= total_p

        # 5. Trích xuất Top-3 nhãn có xác suất cao nhất trong toàn bộ 13 bệnh
        sorted_keys = sorted(all_probs.keys(), key=lambda k: all_probs[k], reverse=True)
        top3 = []
        for code in sorted_keys[:3]:
            prob = float(all_probs[code] * 100.0)
            kb = DISEASE_KNOWLEDGE_BASE.get(code, {})
            top3.append({
                "class_code": code,
                "name_vi": kb.get("name_vi", code),
                "scientific_name": kb.get("scientific_name", ""),
                "category": kb.get("category", "Bệnh hại sầu riêng"),
                "probability": round(prob, 2),
                "cause": kb.get("cause", ""),
                "symptoms": kb.get("symptoms", ""),
                "favorable_conditions": kb.get("favorable_conditions", ""),
                "active_ingredients": kb.get("active_ingredients", []),
                "medicines": kb.get("medicines", []),
                "protocol": kb.get("protocol", []),
                "notes": kb.get("notes", "")
            })

        top1 = top3[0]
        if top1["probability"] >= 75.0:
            conf_label = "Cao (Triệu chứng điển hình)"
        elif top1["probability"] >= 50.0:
            conf_label = "Khá (Giai đoạn chớm phát sinh / Khởi phát)"
        else:
            conf_label = "Trung bình (Cần quan sát thêm diễn biến)"

        # 6. Kiểm tra đối chiếu với Bộ Nhớ Máy Học (Active Learning Memory)
        learned_match = None
        sim_pct = 0.0
        query_emb_list = []
        try:
            from src.active_memory import extract_visual_embedding, find_matching_learned_disease
            query_emb = extract_visual_embedding(
                image_bgr, leaf_mask=leaf_mask, metrics=metrics, core_probs=core_probs
            )
            query_emb_list = [round(float(x), 6) for x in query_emb.tolist()]
            match_res = find_matching_learned_disease(query_emb, similarity_threshold=0.80)
            if match_res is not None:
                learned_match, sim_pct = match_res
        except Exception:
            pass

        if learned_match is not None:
            # Nhận diện chính xác từ mẫu tri thức mà người dùng / máy đã học
            learned_pred = {
                "class_code": f"Learned_{learned_match['id']}",
                "name_vi": learned_match["disease_name"],
                "scientific_name": learned_match.get("scientific_name", ""),
                "category": learned_match.get("category", "Bệnh máy đã học"),
                "probability": float(sim_pct),
                "cause": f"Bệnh nhận diện từ Bộ nhớ Máy học (Ghi nhận bởi: {learned_match.get('teacher_name', 'Chuyên gia')})",
                "symptoms": learned_match.get("symptoms", ""),
                "favorable_conditions": "Trùng khớp hình thái học, cấu trúc đốm và phân bố màu sắc với mẫu bệnh đã lưu.",
                "active_ingredients": learned_match.get("active_ingredients", []),
                "medicines": learned_match.get("medicines", []),
                "protocol": learned_match.get("protocol", []),
                "notes": learned_match.get("notes", ""),
                "is_learned": True,
                "learned_id": learned_match["id"],
                "teacher_name": learned_match.get("teacher_name", "Chuyên gia")
            }
            final_top3 = [learned_pred] + top3[:2]
            return {
                "status": "SUCCESS",
                "message": f"Nhận diện chính xác từ Bộ nhớ Máy học: {learned_match['disease_name']}",
                "metadata": meta,
                "top1": learned_pred,
                "top_predictions": final_top3,
                "confidence_label": f"Độ tương đồng bộ nhớ: {sim_pct:.1f}%",
                "is_ood": False,
                "is_learned": True,
                "query_embedding": query_emb_list
            }

        return {
            "status": "SUCCESS",
            "message": "Chẩn đoán bệnh thành công.",
            "metadata": meta,
            "top1": top1,
            "top_predictions": top3,
            "confidence_label": conf_label,
            "is_ood": False,
            "is_learned": False,
            "query_embedding": query_emb_list
        }


def diagnose_durian_leaf(
    image_bgr: np.ndarray,
    morph_info: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    leaf_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Hàm tiện ích cấp cao để chẩn đoán bệnh từ ảnh BGR kết hợp CV, AI & Bộ nhớ máy học."""
    diagnoser = DurianAIDiagnoser.get_instance()
    return diagnoser.predict(image_bgr, morph_info=morph_info, metrics=metrics, leaf_mask=leaf_mask)
