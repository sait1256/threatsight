"""
ThreatSight - Frame Visualizer
Tespit sonuçlarını frame üzerine çizer.
"""

import cv2
import numpy as np
import time
from src.detector import FrameResult
from src.threat_scorer import ThreatScorer, ThreatAlert, THREAT_COLORS, THREAT_LABELS


class Visualizer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    THICKNESS = 2

    def __init__(self, config: dict):
        disp = config.get("display", {})
        self.show_fps = disp.get("show_fps", True)
        self.show_conf = disp.get("show_confidence", True)
        self.show_threat = disp.get("show_threat_level", True)
        self.bbox_thickness = disp.get("bbox_thickness", 2)

    def draw(self, frame, result: FrameResult, scorer: ThreatScorer,
             active_alert: ThreatAlert = None) -> np.ndarray:
        out = frame.copy()

        for det in result.detections:
            color = scorer.get_frame_color(det.threat_level)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.bbox_thickness)

            label_parts = [det.class_name]
            if self.show_conf:
                label_parts.append(f"{det.confidence:.0%}")
            if self.show_threat:
                label_parts.append(f"L{det.threat_level}")
            label = " | ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 3, y1 - 4),
                self.FONT, self.FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA
            )

        self._draw_hud(out, result, scorer)

        if active_alert:
            self._draw_alert_banner(out, active_alert)

        return out

    def _draw_hud(self, frame, result: FrameResult, scorer: ThreatScorer):
        lines = []
        if self.show_fps:
            lines.append(f"FPS: {result.fps:.1f}  |  {result.frame_time:.1f}ms")
        lines.append(f"Detections: {len(result.detections)}")
        lines.append(f"Alerts: {scorer.alert_count}")
        lines.append(f"Threat: {scorer.get_frame_label(result.max_threat_level)}")

        status_color = THREAT_COLORS.get(result.max_threat_level, (200, 200, 200))

        box_h = len(lines) * 22 + 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            color = status_color if "Threat:" in line else (220, 220, 220)
            cv2.putText(
                frame, line, (8, 20 + i * 22),
                self.FONT, self.FONT_SCALE, color, 1, cv2.LINE_AA
            )

    def _draw_alert_banner(self, frame, alert: ThreatAlert):
        h, w = frame.shape[:2]
        color = alert.color

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        blink = int(time.time() * 2) % 2 == 0
        text_color = (255, 255, 255) if blink else (40, 40, 40)

        cv2.putText(
            frame, f"!! {alert.message}",
            (12, h - 16), self.FONT, 0.65, text_color, 2, cv2.LINE_AA,
        )
        