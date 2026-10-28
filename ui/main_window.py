"""
ThreatSight - PyQt6 Ana Pencere
"""

import sys
import time
import yaml
import numpy as np
import cv2
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QSizePolicy
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont

from src.detector import ThreatDetector, FrameResult
from src.video_stream import VideoStream
from src.threat_scorer import ThreatScorer, ThreatAlert
from src.logger import EventLogger
from src.visualizer import Visualizer


# ── Renkler ──────────────────────────────────────
DARK_BG     = "#1a1a2e"
PANEL_BG    = "#16213e"
CARD_BG     = "#0f3460"
ACCENT      = "#e94560"
GREEN       = "#00b894"
ORANGE      = "#fd9644"
RED         = "#e94560"
TEXT        = "#eaeaea"
TEXT_DIM    = "#888"
BORDER      = "#2a2a4a"


# ── Detection Thread ──────────────────────────────
class DetectionThread(QThread):
    """Kamera okuma ve AI inference ayrı thread'de çalışır."""
    frame_ready   = pyqtSignal(np.ndarray, object)   # (annotated_frame, FrameResult)
    alert_ready   = pyqtSignal(object)               # ThreatAlert
    error_signal  = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config  = config
        self._active = False

    def run(self):
        try:
            detector   = ThreatDetector(self.config)
            detector.warmup()
            scorer     = ThreatScorer(self.config)
            logger     = EventLogger(self.config)
            visualizer = Visualizer(self.config)

            cam = self.config["camera"]
            stream = VideoStream(
                source=cam["source"],
                resolution=tuple(cam["resolution"]),
                fps_limit=cam["fps_limit"],
            ).start()

            self._active = True

            while self._active:
                ret, frame = stream.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                result = detector.detect(frame)
                alert  = scorer.evaluate(result)

                if alert:
                    logger.log_alert(alert, frame)
                    self.alert_ready.emit(alert)

                annotated = visualizer.draw(frame, result, scorer,
                                            alert if alert else None)
                self.frame_ready.emit(annotated, result)

            stream.stop()
            logger.close()

        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._active = False
        self.wait(3000)


# ── Ana Pencere ───────────────────────────────────
class ThreatSightWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._thread = None
        self._start_time = None
        self._fps = 0.0
        self._det_count = 0
        self._alert_count = 0
        self._current_threat = 1

        self.setWindowTitle("ThreatSight — Real-time Threat Detection")
        self.setMinimumSize(1100, 650)
        self._apply_style()
        self._build_ui()

        # Uptime timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_uptime)
        self._timer.start(1000)

    # ── UI Builder ────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Başlık
        title = QLabel("🎯  ThreatSight")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {ACCENT};")
        root.addWidget(title)

        # Ana içerik (kamera + sağ panel)
        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, stretch=1)

        # Kamera
        self.camera_label = QLabel("Kamera başlatılıyor...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(
            f"background:{PANEL_BG}; border:2px solid {BORDER}; border-radius:8px; color:{TEXT_DIM};"
        )
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content.addWidget(self.camera_label, stretch=3)

        # Sağ panel
        right = QVBoxLayout()
        right.setSpacing(8)
        content.addLayout(right, stretch=1)

        # İstatistik kartları
        right.addWidget(self._stat_card())

        # Alert tablosu
        right.addWidget(self._alert_label())
        right.addWidget(self._alert_table(), stretch=1)

        # Alt bar
        root.addWidget(self._bottom_bar())

    def _stat_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"background:{CARD_BG}; border-radius:8px; padding:6px;"
        )
        layout = QVBoxLayout(card)
        layout.setSpacing(4)

        self.lbl_fps     = self._stat_row(layout, "FPS",     "—")
        self.lbl_det     = self._stat_row(layout, "Detections", "0")
        self.lbl_alerts  = self._stat_row(layout, "Alerts",  "0")
        self.lbl_uptime  = self._stat_row(layout, "Uptime",  "00:00")
        return card

    def _stat_row(self, layout, key: str, val: str) -> QLabel:
        row = QHBoxLayout()
        k = QLabel(key)
        k.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        v = QLabel(val)
        v.setStyleSheet(f"color:{TEXT}; font-size:12px; font-weight:bold;")
        v.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(k)
        row.addWidget(v)
        layout.addLayout(row)
        return v

    def _alert_label(self) -> QLabel:
        lbl = QLabel("ALERT GEÇMİŞİ")
        lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; font-weight:bold;")
        return lbl

    def _alert_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Saat", "Seviye", "Mesaj"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background:{PANEL_BG}; color:{TEXT};
                gridline-color:{BORDER}; border:none; font-size:11px;
            }}
            QTableWidget::item {{ padding:4px; }}
        """)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setColumnWidth(0, 55)
        self.table.setColumnWidth(1, 55)
        return self.table

    def _bottom_bar(self) -> QFrame:
        bar = QFrame()
        bar.setStyleSheet(
            f"background:{PANEL_BG}; border-radius:8px; padding:4px;"
        )
        layout = QHBoxLayout(bar)
        layout.setSpacing(10)

        self.btn_start = QPushButton("▶  BAŞLAT")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_start.setStyleSheet(self._btn_style(GREEN))

        self.btn_stop = QPushButton("■  DURDUR")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_style(RED))

        self.lbl_threat_bar = QLabel("Threat: —")
        self.lbl_threat_bar.setStyleSheet(
            f"color:{TEXT}; font-size:13px; font-weight:bold;"
        )
        self.lbl_threat_bar.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addStretch()
        layout.addWidget(self.lbl_threat_bar)
        return bar

    def _btn_style(self, color: str) -> str:
        return f"""
            QPushButton {{
                background:{color}; color:white;
                border:none; border-radius:6px;
                padding:8px 20px; font-size:13px; font-weight:bold;
            }}
            QPushButton:hover {{ background:{color}cc; }}
            QPushButton:disabled {{ background:#333; color:#666; }}
        """

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{DARK_BG}; color:{TEXT}; }}
            QLabel {{ color:{TEXT}; }}
        """)

    # ── Slots ─────────────────────────────────────
    def start_detection(self):
        self._thread = DetectionThread(self.config)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.alert_ready.connect(self._on_alert)
        self._thread.error_signal.connect(self._on_error)
        self._thread.start()

        self._start_time = time.time()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_detection(self):
        if self._thread:
            self._thread.stop()
            self._thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_label.setText("Durduruldu.")
        self.lbl_threat_bar.setText("Threat: —")
        self.lbl_threat_bar.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:bold;")

    def _on_frame(self, frame: np.ndarray, result):
        self._fps = result.fps
        self._det_count = len(result.detections)
        self._current_threat = result.max_threat_level

        # Frame'i Qt pixmap'e çevir
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(pix)

        # Stats güncelle
        self.lbl_fps.setText(f"{result.fps:.1f}")
        self.lbl_det.setText(str(self._det_count))
        self.lbl_alerts.setText(str(self._alert_count))

        # Threat bar rengi
        colors = {1: GREEN, 2: ORANGE, 3: RED}
        labels = {1: "LOW 🟢", 2: "MEDIUM 🟠", 3: "HIGH 🔴"}
        c = colors.get(self._current_threat, TEXT)
        l = labels.get(self._current_threat, "—")
        self.lbl_threat_bar.setText(f"Threat: {l}")
        self.lbl_threat_bar.setStyleSheet(
            f"color:{c}; font-size:13px; font-weight:bold;"
        )

    def _on_alert(self, alert: ThreatAlert):
        self._alert_count += 1
        self.lbl_alerts.setText(str(self._alert_count))

        colors = {1: GREEN, 2: ORANGE, 3: RED}
        color  = colors.get(alert.level, TEXT)

        row = self.table.rowCount()
        self.table.insertRow(row)

        saat = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
        items = [
            QTableWidgetItem(saat),
            QTableWidgetItem(alert.label),
            QTableWidgetItem(alert.message),
        ]
        for col, item in enumerate(items):
            item.setForeground(QColor(color))
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()

    def _on_error(self, msg: str):
        self.camera_label.setText(f"HATA: {msg}")

    def _update_uptime(self):
        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            m, s = divmod(elapsed, 60)
            self.lbl_uptime.setText(f"{m:02d}:{s:02d}")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()