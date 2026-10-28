# =============================================================
# ThreatSight — Dockerfile
# =============================================================
# Aşama 1: Bağımlılıkları kur (builder)
# Aşama 2: Sadece çalışma zamanını kopyala (runtime)
# Sonuç: ~2.5 GB → ~1.8 GB (daha küçük final imaj)
# =============================================================

# --- Builder aşaması ---
FROM python:3.10-slim AS builder

WORKDIR /install

# Sistem kütüphaneleri (OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (GPU yoksa ~800 MB tasarruf)
# GPU için: torch torchvision (--extra-index-url kaldır)
RUN pip install --upgrade pip && \
    pip install --prefix=/install/pkg --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Uygulama bağımlılıkları
COPY requirements.txt .
RUN pip install --prefix=/install/pkg --no-cache-dir \
        fastapi>=0.110.0 \
        "uvicorn[standard]>=0.27.0" \
        python-multipart>=0.0.9 \
        ultralytics>=8.0.0 \
        opencv-python-headless>=4.8.0 \
        PyYAML>=6.0 \
        numpy>=1.24.0


# --- Runtime aşaması ---
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="ThreatSight"
LABEL org.opencontainers.image.description="Gerçek zamanlı AI tehdit tespit sistemi"
LABEL org.opencontainers.image.source="https://github.com/sait1256/threatsight"

# Runtime sistem kütüphaneleri (OpenCV headless için)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Builder'dan kurulan paketleri kopyala
COPY --from=builder /install/pkg /usr/local

WORKDIR /app

# Uygulama kodunu kopyala
COPY config/     ./config/
COPY src/        ./src/
COPY scripts/    ./scripts/
COPY frontend/   ./frontend/

# Port
EXPOSE 8000

# Sağlık kontrolü — her 30 sn /health endpoint'ini kontrol et
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Kamera erişimi için root olmayan kullanıcı (video grubunda)
RUN useradd -m -u 1000 threatsight && \
    usermod -aG video threatsight 2>/dev/null || true
USER threatsight

# Başlatma komutu
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
