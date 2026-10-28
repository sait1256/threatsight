"""
ThreatSight - Unit Tests: ThreatScorer
Tehdit puanlama motorunun doğru çalışıp çalışmadığını test eder.
"""

import time
import pytest
from dataclasses import dataclass
from typing import List


# ── Sahte Detection nesnesi (gerçek model olmadan test için) ──────────────────
@dataclass
class FakeDetection:
    class_name: str
    confidence: float
    bbox: tuple
    bbox_norm: tuple
    threat_level: int
    timestamp: float = 0.0


@dataclass
class FakeFrameResult:
    detections: List[FakeDetection]
    frame_time: float = 10.0
    fps: float = 30.0
    max_threat_level: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.detections:
            self.max_threat_level = max(d.threat_level for d in self.detections)


# ── Config yardımcısı ─────────────────────────────────────────────────────────
def make_config(cooldown=0.0, threshold=3, enabled=True):
    return {
        "alerts": {
            "enabled": enabled,
            "cooldown_seconds": cooldown,
            "high_threat_threshold": threshold,
        },
        "threat": {
            "multi_threat_multiplier": 1.5,
        },
    }


# ── Testler ───────────────────────────────────────────────────────────────────
class TestThreatScorer:

    def _scorer(self, **kwargs):
        from src.threat_scorer import ThreatScorer
        return ThreatScorer(make_config(**kwargs))

    def test_no_detections_returns_none(self):
        """Tespit yoksa alert üretilmemeli."""
        scorer = self._scorer()
        result = FakeFrameResult(detections=[])
        assert scorer.evaluate(result) is None

    def test_low_threat_returns_none(self):
        """Düşük tehdit (level 1) alert üretmemeli."""
        scorer = self._scorer()
        det = FakeDetection("person", 0.9, (0,0,100,100), (0,0,1,1), threat_level=1)
        result = FakeFrameResult(detections=[det])
        assert scorer.evaluate(result) is None

    def test_high_threat_returns_alert(self):
        """Yüksek tehdit (level 3) alert üretmeli."""
        scorer = self._scorer(cooldown=0.0)
        det = FakeDetection("knife", 0.95, (0,0,50,50), (0,0,1,1), threat_level=3)
        result = FakeFrameResult(detections=[det])
        alert = scorer.evaluate(result)
        assert alert is not None
        assert alert.level == 3
        assert alert.label == "HIGH"

    def test_cooldown_blocks_second_alert(self):
        """Cooldown süresi içinde ikinci alert üretilmemeli."""
        scorer = self._scorer(cooldown=10.0)
        det = FakeDetection("gun", 0.9, (0,0,50,50), (0,0,1,1), threat_level=3)
        result = FakeFrameResult(detections=[det])

        first = scorer.evaluate(result)
        second = scorer.evaluate(result)

        assert first is not None
        assert second is None  # cooldown nedeniyle bloklandı

    def test_multiple_threats_message(self):
        """Birden fazla tehdit varsa mesaj 'MULTIPLE' içermeli."""
        scorer = self._scorer(cooldown=0.0)
        dets = [
            FakeDetection("knife", 0.9, (0,0,50,50), (0,0,1,1), threat_level=3),
            FakeDetection("gun",   0.8, (50,50,100,100), (0,0,1,1), threat_level=3),
        ]
        result = FakeFrameResult(detections=dets)
        alert = scorer.evaluate(result)
        assert alert is not None
        assert "MULTIPLE" in alert.message

    def test_alerts_disabled(self):
        """Alert sistemi kapalıysa hiç alert üretilmemeli."""
        scorer = self._scorer(enabled=False)
        det = FakeDetection("knife", 0.99, (0,0,50,50), (0,0,1,1), threat_level=3)
        result = FakeFrameResult(detections=[det])
        assert scorer.evaluate(result) is None

    def test_alert_count_increments(self):
        """Her alert sonrası sayaç artmalı."""
        scorer = self._scorer(cooldown=0.0)
        det = FakeDetection("drone", 0.85, (0,0,50,50), (0,0,1,1), threat_level=3)
        result = FakeFrameResult(detections=[det])

        assert scorer.alert_count == 0
        scorer.evaluate(result)
        assert scorer.alert_count == 1

    def test_alert_color_is_red_for_high(self):
        """Yüksek tehdit rengi kırmızı (BGR: 0,0,220) olmalı."""
        scorer = self._scorer(cooldown=0.0)
        det = FakeDetection("knife", 0.9, (0,0,50,50), (0,0,1,1), threat_level=3)
        result = FakeFrameResult(detections=[det])
        alert = scorer.evaluate(result)
        assert alert.color == (0, 0, 220)
