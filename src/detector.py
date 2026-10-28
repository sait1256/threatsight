"""
ThreatSight - Core Detection Engine
YOLOv8 tabanlı GPU-accelerated nesne tespit motoru.
"""

import torch
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List
import time


@dataclass
class Detection:
    """Tek bir tespit sonucunu temsil eder."""
    class_name: str
    confidence: float
    bbox: tuple          # (x1, y1, x2, y2) pixel koordinatları
    bbox_norm: tuple     # Normalize edilmiş koordinatlar (0-1)
    threat_level: int    # 1=Düşük, 2=Orta, 3=Yüksek
    timestamp: float


@dataclass
class FrameResult:
    """Bir frame'deki tüm tespit sonuçlarını içerir."""
    detections: List[Detection]
    frame_time: float
    fps: float
    max_threat_level: int
    timestamp: float


class ThreatDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config["model"]
        self.threat_cfg = config["threat"]

        self.threat_levels = self.threat_cfg.get("levels", {})
        self.device = self._select_device(self.model_cfg.get("device", "cuda"))

        print(f"[ThreatDetector] Model yükleniyor: {self.model_cfg['weights']}")
        self.model = YOLO(self.model_cfg["weights"])
        self.model.to(self.device)

        self.conf_threshold = self.model_cfg.get("confidence", 0.45)
        self.iou_threshold = self.model_cfg.get("iou_threshold", 0.50)
        self.img_size = self.model_cfg.get("image_size", 640)

        self._fps_buffer: List[float] = []
        print(f"[ThreatDetector] Hazır. Device: {self.device}")

    def _select_device(self, preferred: str) -> str:
        if preferred == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[ThreatDetector] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            return "cuda"
        else:
            if preferred == "cuda":
                print("[ThreatDetector] UYARI: CUDA bulunamadı, CPU kullanılıyor.")
            return "cpu"

    def _get_threat_level(self, class_name: str) -> int:
        return self.threat_levels.get(class_name.lower(), 1)

    def _calculate_fps(self, elapsed_ms: float) -> float:
        self._fps_buffer.append(1000.0 / max(elapsed_ms, 1))
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        return sum(self._fps_buffer) / len(self._fps_buffer)

    def detect(self, frame: np.ndarray) -> FrameResult:
        start_time = time.perf_counter()
        timestamp = time.time()
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        max_threat = 1

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox_norm = (x1/w, y1/h, x2/w, y2/h)
                threat_level = self._get_threat_level(class_name)
                max_threat = max(max_threat, threat_level)

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    bbox_norm=bbox_norm,
                    threat_level=threat_level,
                    timestamp=timestamp
                ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        fps = self._calculate_fps(elapsed_ms)

        return FrameResult(
            detections=detections,
            frame_time=elapsed_ms,
            fps=fps,
            max_threat_level=max_threat,
            timestamp=timestamp
        )

    def warmup(self):
        """Modeli ısıt — ilk frame gecikmesini önler."""
        print("[ThreatDetector] Model ısıtılıyor...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.detect(dummy)
        print("[ThreatDetector] Hazır.")
