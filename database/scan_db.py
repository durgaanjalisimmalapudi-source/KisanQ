"""
Offline SQLite Database for KisanQ.
Persists scan history, crop profiles, disease diagnoses, confidence, sensor readings, and Telugu audio paths.
"""

import sys
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class ScanDatabase:
    """Manages local SQLite database for offline scan persistence."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else config.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Create required tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    image_path TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_uncertain INTEGER NOT NULL,
                    quantum_measurements TEXT,
                    classical_latency_ms REAL,
                    quantum_latency_ms REAL,
                    temp_c REAL,
                    humidity_pct REAL,
                    soil_moisture_pct REAL,
                    microclimate_risk TEXT,
                    telugu_summary TEXT,
                    english_summary TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crop_catalog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crop_name TEXT UNIQUE NOT NULL,
                    botanical_name TEXT,
                    typical_diseases TEXT
                )
            """)
            # Seed default mango profile
            cursor.execute("""
                INSERT OR IGNORE INTO crop_catalog (crop_name, botanical_name, typical_diseases)
                VALUES (
                    'Mango (మామిడి)',
                    'Mangifera indica',
                    'Anthracnose, Bacterial Canker, Powdery Mildew, Gall Midge, Die Back'
                )
            """)
            conn.commit()

    def record_scan(
        self,
        crop_name: str,
        predicted_class: str,
        confidence: float,
        is_uncertain: bool,
        image_path: Optional[str] = None,
        quantum_measurements: Optional[List[float]] = None,
        classical_latency_ms: float = 0.0,
        quantum_latency_ms: float = 0.0,
        temp_c: float = 28.0,
        humidity_pct: float = 75.0,
        soil_moisture_pct: float = 50.0,
        microclimate_risk: str = "Moderate",
        telugu_summary: str = "",
        english_summary: str = ""
    ) -> int:
        """Insert a completed scan record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scans (
                    timestamp, crop_name, image_path, predicted_class, confidence,
                    is_uncertain, quantum_measurements, classical_latency_ms, quantum_latency_ms,
                    temp_c, humidity_pct, soil_moisture_pct, microclimate_risk,
                    telugu_summary, english_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                crop_name,
                str(image_path) if image_path else "",
                predicted_class,
                confidence,
                1 if is_uncertain else 0,
                json.dumps(quantum_measurements or []),
                classical_latency_ms,
                quantum_latency_ms,
                temp_c,
                humidity_pct,
                soil_moisture_pct,
                microclimate_risk,
                telugu_summary,
                english_summary
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_scans(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieve recent scans formatted as dictionaries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


if __name__ == "__main__":
    db = ScanDatabase()
    scan_id = db.record_scan(
        crop_name="Mango",
        predicted_class="Healthy",
        confidence=98.5,
        is_uncertain=False,
        quantum_measurements=[0.12, -0.45, 0.88, -0.11]
    )
    print("Recorded test scan with ID:", scan_id)
    recent = db.get_recent_scans(limit=3)
    print("Recent scans:", len(recent))
