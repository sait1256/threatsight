"""
ThreatSight - Stream Manager
============================
Arka planda kamera + YOLOv8 inference çalıştırır.
Sonuçları thread-safe olarak FastAPI'ye sunar.
"""

import threading
import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class Detection:
    label: str
    confidence: float
    threat_level: str
    bbox: list  # [x1, y1, x2, y2]


@dataclass
class FrameState:
    frame_jpg: Optional[bytes] = None
    detections: list = field(default_factory=list)
    fps: float = 0.0
    frame_count: int = 0
    uptime_sec: float = 0.0
    running: bool = False


# Tehdit seviyeleri (threat_scorer ile uyumlu)
THREAT_COLORS = {
    "CRITICAL": (0, 0, 255),    # Kırmızı
    "HIGH":     (0, 100, 255),  # Turuncu
    "MEDIUM":   (0, 200, 255),  # Sarı
    "LOW":      (0, 255, 0),    # Yeşil
    "SAFE":     (200, 200, 200),
}

THREAT_MAP = {
    "person":   ("LOW",      0.30),
    "knife":    ("CRITICAL", 0.75),
    "scissors": ("HIGH",     0.55),
    "cell phone": ("LOW",    0.20),
    "backpack": ("LOW",      0.15),
    "car":      ("LOW",      0.10),
    "truck":    ("MEDIUM",   0.35),
}


class StreamManager:
    """
    Kamera + YOLOv8 inference'ı arka planda çalıştıran sınıf.
    FastAPI thread'leri state'i güvenle okuyabilir.
    """

    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._state = FrameState()
        self._alert_history: deque = deque(maxlen=50)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Config
        self.source = config.get("camera", {}).get("source", 0)
        self.model_path = config.get("model", {}).get("path", "yolov8n.pt")
        self.conf_thresh = config.get("model", {}).get("confidence", 0.4)
        self.img_size = config.get("model", {}).get("img_size", 640)
        self.alert_threshold = config.get("threat", {}).get("alert_threshold", 0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Inference thread'ini başlat."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Inference thread'ini durdur."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_frame_jpg(self) -> Optional[bytes]:
        with self._lock:
            return self._state.frame_jpg

    def get_detections(self) -> list:
        with self._lock:
            return list(self._state.detections)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "fps": round(self._state.fps, 1),
                "frame_count": self._state.frame_count,
                "uptime_sec": round(self._state.uptime_sec, 1),
                "running": self._state.running,
                "source": str(self.source),
                "model": str(self.model_path),
            }

    def get_alerts(self) -> list:
        with self._lock:
            return list(self._alert_history)

    def update_config(self, conf: float = None, alert_threshold: float = None):
        if conf is not None:
            self.conf_thresh = conf
        if alert_threshold is not None:
            self.alert_threshold = alert_threshold

    # ------------------------------------------------------------------
    # Private — inference döngüsü
    # ------------------------------------------------------------------

    def _run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
        except Exception as e:
            print(f"[StreamManager] Model yüklenemedi: {e}")
            return

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            # Kamera yoksa demo modu — siyah frame gönder
            self._demo_mode()
            return

        start_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        with self._lock:
            self._state.running = True

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # Inference
                results = model(
                    frame,
                    conf=self.conf_thresh,
                    imgsz=self.img_size,
                    verbose=False
                )

                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    threat_level, _ = THREAT_MAP.get(label, ("SAFE", 0.0))

                    detections.append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "threat_level": threat_level,
                        "bbox": [int(v) for v in xyxy],
                    })

                    # Alert kaydı
                    if threat_level in ("CRITICAL", "HIGH"):
                        alert = {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "label": label,
                            "confidence": round(conf, 3),
                            "threat_level": threat_level,
                        }
                        with self._lock:
                            self._alert_history.appendleft(alert)

                # Frame'i annotate et
                annotated = self._annotate(frame.copy(), detections)
                _, jpg = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                fps_counter += 1
                now = time.time()
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_timer = now
                else:
                    fps = self._state.fps  # önceki değeri koru

                with self._lock:
                    self._state.frame_jpg = jpg.tobytes()
                    self._state.detections = detections
                    self._state.fps = fps
                    self._state.frame_count += 1
                    self._state.uptime_sec = now - start_time

        finally:
            cap.release()
            with self._lock:
                self._state.running = False

    def _demo_mode(self):
        """Kamera yokken demo frame üretir."""
        print("[StreamManager] Kamera bulunamadı — demo modunda çalışıyor")
        start_time = time.time()
        frame_count = 0

        with self._lock:
            self._state.running = True

        while not self._stop_event.is_set():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame, "ThreatSight DEMO", (160, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 100), 2)
            cv2.putText(frame, f"Kamera yok | {ts}", (170, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (250, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 200), 1)

            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_count += 1

            with self._lock:
                self._state.frame_jpg = jpg.tobytes()
                self._state.fps = 10.0
                self._state.frame_count = frame_count
                self._state.uptime_sec = time.time() - start_time

            time.sleep(0.1)

        with self._lock:
            self._state.running = False

    def _annotate(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = THREAT_COLORS.get(det["threat_level"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det['label']} {det['confidence']:.2f} [{det['threat_level']}]"
            cv2.putText(frame, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # FPS overlay
        fps_text = f"FPS: {self._state.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame


def load_stream_manager(config_path: str = "config/config.yaml") -> StreamManager:
    """Config'den StreamManager oluştur."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return StreamManager(config)
