"""
ThreatSight - Event Logger
SQLite tabanlı olay ve tespit kayıt sistemi.
"""

import sqlite3
import json
import time
import os
import cv2
import numpy as np
from typing import List, Optional
from src.detector import Detection
from src.threat_scorer import ThreatAlert


class EventLogger:
    def __init__(self, config: dict):
        log_cfg = config.get("logging", {})
        self.enabled = log_cfg.get("enabled", True)
        self.db_path = log_cfg.get("db_path", "logs/events.db")
        self.save_snapshots = log_cfg.get("save_snapshots", True)
        self.snapshot_dir = log_cfg.get("snapshot_dir", "logs/snapshots")

        if self.enabled:
            self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if self.save_snapshots:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                level       INTEGER NOT NULL,
                label       TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                detections  TEXT,
                snapshot    TEXT
            );

            CREATE TABLE IF NOT EXISTS detections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    REAL    NOT NULL,
                class_name   TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                threat_level INTEGER NOT NULL,
                bbox_x1      INTEGER,
                bbox_y1      INTEGER,
                bbox_x2      INTEGER,
                bbox_y2      INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp);
        """)
        self._conn.commit()
        print(f"[EventLogger] Veritabanı: {self.db_path}")

    def log_alert(self, alert: ThreatAlert, frame: Optional[np.ndarray] = None) -> int:
        if not self.enabled:
            return -1

        snapshot_path = None
        if self.save_snapshots and frame is not None:
            snapshot_path = self._save_snapshot(frame, alert)

        detections_json = json.dumps([
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "threat_level": d.threat_level,
                "bbox": d.bbox,
            }
            for d in alert.detections
        ])

        cursor = self._conn.execute(
            """INSERT INTO alerts (timestamp, level, label, message, detections, snapshot)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (alert.timestamp, alert.level, alert.label,
             alert.message, detections_json, snapshot_path)
        )
        self._conn.commit()
        return cursor.lastrowid

    def _save_snapshot(self, frame: np.ndarray, alert: ThreatAlert) -> str:
        ts = int(alert.timestamp)
        filename = f"alert_{ts}_L{alert.level}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    def get_stats(self) -> dict:
        if not self.enabled:
            return {}
        total = self._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        high = self._conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE level >= 3"
        ).fetchone()[0]
        return {
            "total_alerts": total,
            "high_threat_alerts": high,
            "db_path": self.db_path,
        }

    def close(self):
        if self.enabled and hasattr(self, "_conn"):
            self._conn.close()
            