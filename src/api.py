"""
ThreatSight - FastAPI REST API
===============================
Endpoint listesi:
  GET  /               → sağlık kontrolü
  GET  /stream         → MJPEG canlı video
  GET  /detections     → son tespitler (JSON)
  GET  /alerts         → alert geçmişi (JSON)
  GET  /stats          → sistem istatistikleri (JSON)
  POST /config         → eşik değerlerini güncelle
  GET  /docs           → Swagger UI (otomatik)

Kullanım:
    python scripts/run_api.py
    # veya
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.stream_manager import load_stream_manager

# ------------------------------------------------------------------
# Uygulama başlatma
# ------------------------------------------------------------------

app = FastAPI(
    title="ThreatSight API",
    description="Gerçek zamanlı AI tehdit tespit sistemi REST API",
    version="1.0.0",
)

# CORS — frontend farklı port'tan erişebilsin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# StreamManager — uygulama başlarken tek instance oluştur
_manager = None


@app.on_event("startup")
async def startup():
    global _manager
    _manager = load_stream_manager("config/config.yaml")
    _manager.start()
    print("[API] StreamManager başlatıldı")


@app.on_event("shutdown")
async def shutdown():
    global _manager
    if _manager:
        _manager.stop()
        print("[API] StreamManager durduruldu")


# ------------------------------------------------------------------
# Static dosyalar (frontend)
# ------------------------------------------------------------------

FRONTEND_DIR = Path("frontend")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ------------------------------------------------------------------
# Endpoint'ler
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Genel"])
async def root():
    """
    Ana sayfa — frontend dashboard'u döner.
    frontend/index.html varsa onu, yoksa basit JSON döner.
    """
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("""
    <html><body>
    <h2>ThreatSight API çalışıyor ✅</h2>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/stream">MJPEG Stream</a></li>
      <li><a href="/detections">Tespitler</a></li>
      <li><a href="/alerts">Alertler</a></li>
      <li><a href="/stats">İstatistikler</a></li>
    </ul>
    </body></html>
    """)


@app.get("/health", tags=["Genel"])
async def health():
    """Sistem sağlık kontrolü."""
    return {"status": "ok", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.get("/stream", tags=["Video"])
async def video_stream():
    """
    MJPEG canlı video akışı.
    <img src="http://localhost:8000/stream"> ile sayfaya göm.
    """
    async def generate():
        while True:
            jpg = _manager.get_frame_jpg() if _manager else None
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpg +
                    b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30 FPS hedef

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detections", tags=["Tespit"])
async def get_detections():
    """Son frame'deki tespitler."""
    detections = _manager.get_detections() if _manager else []
    return {
        "count": len(detections),
        "detections": detections,
        "timestamp": time.strftime("%H:%M:%S"),
    }


@app.get("/alerts", tags=["Tespit"])
async def get_alerts(limit: int = 20):
    """
    Son alert geçmişi.
    ?limit=N ile kaç kayıt istediğini belirt (max 50).
    """
    alerts = _manager.get_alerts() if _manager else []
    return {
        "count": len(alerts),
        "alerts": alerts[:min(limit, 50)],
    }


@app.get("/stats", tags=["Sistem"])
async def get_stats():
    """Sistem istatistikleri: FPS, uptime, model bilgisi."""
    stats = _manager.get_stats() if _manager else {}
    return stats


class ConfigUpdate(BaseModel):
    confidence: Optional[float] = None      # 0.1 – 0.9
    alert_threshold: Optional[float] = None  # 0.1 – 0.9


@app.post("/config", tags=["Sistem"])
async def update_config(body: ConfigUpdate):
    """
    Çalışırken eşik değerlerini güncelle.
    Örnek: {"confidence": 0.5, "alert_threshold": 0.6}
    """
    if not _manager:
        return {"error": "Manager hazır değil"}

    updates = {}
    if body.confidence is not None:
        if not 0.1 <= body.confidence <= 0.9:
            return {"error": "confidence 0.1-0.9 arasında olmalı"}
        updates["conf"] = body.confidence

    if body.alert_threshold is not None:
        if not 0.1 <= body.alert_threshold <= 0.9:
            return {"error": "alert_threshold 0.1-0.9 arasında olmalı"}
        updates["alert_threshold"] = body.alert_threshold

    _manager.update_config(**updates)
    return {"status": "güncellendi", "yeni_degerler": updates}
