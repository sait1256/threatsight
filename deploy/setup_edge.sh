#!/usr/bin/env bash
# ThreatSight — Raspberry Pi / Jetson Nano kurulum scripti
# Kullanım: bash deploy/setup_edge.sh
set -e

echo "╔══════════════════════════════════════╗"
echo "║  ThreatSight Edge Kurulum            ║"
echo "╚══════════════════════════════════════╝"

# ─── Sistem paketleri ───────────────────────────────────────────────────────
echo "[1/6] Sistem paketleri güncelleniyor..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv \
    libopencv-dev python3-opencv \
    mosquitto-clients \
    libatlas-base-dev   # NumPy için

# ─── Python sanal ortamı ────────────────────────────────────────────────────
echo "[2/6] Python sanal ortamı oluşturuluyor..."
python3 -m venv venv
source venv/bin/activate

# ─── Python bağımlılıkları ──────────────────────────────────────────────────
echo "[3/6] Python bağımlılıkları kuruluyor..."
pip install --upgrade pip -q
pip install -q \
    onnxruntime \
    opencv-python-headless \
    numpy \
    paho-mqtt

# Jetson Nano ise onnxruntime yerine onnxruntime-gpu dene
if [ "$(uname -m)" = "aarch64" ]; then
    pip install -q onnxruntime-gpu 2>/dev/null || true
fi

# ─── Klasör yapısı ──────────────────────────────────────────────────────────
echo "[4/6] Klasör yapısı oluşturuluyor..."
mkdir -p models logs

# ─── Model kopyala ──────────────────────────────────────────────────────────
echo "[5/6] ONNX model kontrolü..."
if [ ! -f "models/threat_v1_best.onnx" ]; then
    echo "  ⚠ models/threat_v1_best.onnx bulunamadı."
    echo "  Ana bilgisayardan kopyala:"
    echo "  scp kullanici@pc:/path/to/threatsight/models/threat_v1_best.onnx models/"
else
    echo "  ✓ Model mevcut"
fi

# ─── Systemd servis ──────────────────────────────────────────────────────────
echo "[6/6] Systemd servisi kuruluyor..."
sudo cp deploy/threatsight.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable threatsight.service
echo "  ✓ Servis aktif edildi (henüz başlatılmadı)"

echo ""
echo "═══════════════════════════════════════"
echo " Kurulum tamamlandı!"
echo "═══════════════════════════════════════"
echo ""
echo "Servisi başlatmak için:"
echo "  sudo systemctl start threatsight"
echo ""
echo "Logları izlemek için:"
echo "  journalctl -u threatsight -f"
echo ""
echo "Manuel test için:"
echo "  source venv/bin/activate"
echo "  python deploy/edge_inference.py --headless --conf 0.4"
