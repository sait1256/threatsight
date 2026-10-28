"""
ThreatSight - Unit Tests: Config
Config dosyasının doğru yüklenip yüklenmediğini test eder.
"""

import yaml
import pytest
from pathlib import Path


CONFIG_PATH = Path("config/config.yaml")


@pytest.fixture
def config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestConfig:

    def test_config_file_exists(self):
        """Config dosyası mevcut olmalı."""
        assert CONFIG_PATH.exists(), "config/config.yaml bulunamadı"

    def test_required_sections(self, config):
        """Zorunlu bölümler mevcut olmalı."""
        for section in ["model", "camera", "threat", "alerts", "logging", "display"]:
            assert section in config, f"'{section}' bölümü eksik"

    def test_model_weights_defined(self, config):
        """Model weights tanımlı olmalı."""
        assert config["model"]["weights"] is not None

    def test_confidence_in_range(self, config):
        """Confidence 0-1 arasında olmalı."""
        conf = config["model"]["confidence"]
        assert 0.0 < conf < 1.0, f"Confidence geçersiz: {conf}"

    def test_threat_levels_valid(self, config):
        """Tüm tehdit seviyeleri 1-3 arasında olmalı."""
        for cls, level in config["threat"]["levels"].items():
            assert 1 <= level <= 3, f"{cls} için geçersiz tehdit seviyesi: {level}"

    def test_camera_source_defined(self, config):
        """Kamera kaynağı tanımlı olmalı."""
        assert "source" in config["camera"]

    def test_alert_cooldown_positive(self, config):
        """Alert cooldown pozitif olmalı."""
        assert config["alerts"]["cooldown_seconds"] > 0

    def test_high_threat_threshold_is_3(self, config):
        """Yüksek tehdit eşiği 3 olmalı."""
        assert config["alerts"]["high_threat_threshold"] == 3

    def test_resolution_has_two_values(self, config):
        """Çözünürlük [genişlik, yükseklik] formatında olmalı."""
        res = config["camera"]["resolution"]
        assert len(res) == 2
        assert res[0] > 0 and res[1] > 0
