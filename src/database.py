"""
Module Quản Lý Cơ Sở Dữ Liệu SQLite Lưu Lịch Sử Chẩn Đoán Lá Sầu Riêng
Không yêu cầu cài đặt máy chủ ngoài, an toàn, hỗ trợ giao dịch ACID.
"""

import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DB_DIR / "diagnosis_history.db"


def get_db_connection() -> sqlite3.Connection:
    """Tạo hoặc mở kết nối tới cơ sở dữ liệu SQLite."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Khởi tạo bảng lưu trữ lịch sử chẩn đoán nếu chưa tồn tại."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT NOT NULL,
                disease_code TEXT NOT NULL,
                disease_name_vi TEXT NOT NULL,
                scientific_name TEXT NOT NULL,
                ai_probability REAL NOT NULL,
                severity_percentage REAL,
                severity_grade TEXT,
                model_version TEXT NOT NULL,
                user_note TEXT,
                status TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_timestamp ON diagnosis_history(timestamp DESC);
        """)
    conn.close()


def save_diagnosis(
    image_name: str,
    disease_code: str,
    disease_name_vi: str,
    scientific_name: str,
    ai_probability: float,
    severity_percentage: Optional[float] = None,
    severity_grade: Optional[str] = None,
    model_version: str = "MobileNetV3-Durian-v1.0",
    user_note: str = "",
    status: str = "SUCCESS"
) -> int:
    """Lưu một bản ghi kết quả chẩn đoán vào cơ sở dữ liệu."""
    init_database()
    conn = get_db_connection()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        cur = conn.execute("""
            INSERT INTO diagnosis_history (
                timestamp, image_name, disease_code, disease_name_vi,
                scientific_name, ai_probability, severity_percentage,
                severity_grade, model_version, user_note, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str, image_name, disease_code, disease_name_vi,
            scientific_name, ai_probability, severity_percentage,
            severity_grade, model_version, user_note, status
        ))
        record_id = cur.lastrowid
    conn.close()
    return record_id


def get_recent_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Lấy danh sách các lần chẩn đoán gần nhất."""
    init_database()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM diagnosis_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_history():
    """Xóa sạch lịch sử chẩn đoán."""
    init_database()
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM diagnosis_history")
    conn.close()
