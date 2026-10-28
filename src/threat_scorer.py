"""
ThreatSight - Threat Scoring Engine
Tespitlere göre tehdit seviyesi hesaplar ve alert üretir.
"""

import time
from dataclasses import dataclass
from typing import List, Optional
from src.detector import Detection, FrameResult


THREAT_COLORS = {
    1: (0, 200, 0),      # Yeşil — Düşük
    2: (0, 165, 255),    # Turuncu — Orta
    3: (0, 0, 220),      # Kırmızı — Yüksek
}

THREAT_LABELS = {
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
}


@dataclass
class ThreatAlert:
    """Tetiklenen bir tehdit uyarısını temsil eder."""
    level: int
    label: str
    detections: List[Detection]
    message: str
    timestamp: float
    color: tuple


class ThreatScorer:
    def __init__(self, config: dict):
        self.alert_cfg = config.get("alerts", {})
        self.threat_cfg = config.get("threat", {})

        self.alerts_enabled = self.alert_cfg.get("enabled", True)
        self.cooldown = self.alert_cfg.get("cooldown_seconds", 5.0)
        self.high_threshold = self.alert_cfg.get("high_threat_threshold", 3)

        self._last_alert_time = 0.0
        self._alert_history: List[ThreatAlert] = []

    def evaluate(self, result: FrameResult) -> Optional[ThreatAlert]:
        """Frame sonucunu değerlendir, gerekiyorsa alert döndür."""
        if not result.detections:
            return None

        high_threats = [d for d in result.detections if d.threat_level >= self.high_threshold]

        if high_threats and self.alerts_enabled:
            now = time.time()
            if now - self._last_alert_time >= self.cooldown:
                alert = self._create_alert(high_threats, result.max_threat_level)
                self._last_alert_time = now
                self._alert_history.append(alert)
                return alert

        return None

    def _create_alert(self, threats: List[Detection], max_level: int) -> ThreatAlert:
        count = len(threats)
        names = list({d.class_name for d in threats})

        if count > 1:
            message = f"MULTIPLE THREATS: {', '.join(names)} ({count} targets)"
        else:
            message = f"THREAT DETECTED: {names[0].upper()} (conf: {threats[0].confidence:.0%})"

        return ThreatAlert(
            level=max_level,
            label=THREAT_LABELS.get(max_level, "UNKNOWN"),
            detections=threats,
            message=message,
            timestamp=time.time(),
            color=THREAT_COLORS.get(max_level, (255, 255, 255)),
        )

    def get_frame_color(self, threat_level: int) -> tuple:
        return THREAT_COLORS.get(threat_level, (200, 200, 200))

    def get_frame_label(self, threat_level: int) -> str:
        return THREAT_LABELS.get(threat_level, "UNKNOWN")

    @property
    def alert_count(self) -> int:
        return len(self._alert_history)