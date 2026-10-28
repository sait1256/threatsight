"""
ThreatSight - API Launcher
===========================
FastAPI sunucusunu başlatır.

Kullanım:
    python scripts/run_api.py                   # localhost:8000
    python scripts/run_api.py --host 0.0.0.0   # ağdan erişilebilir
    python scripts/run_api.py --port 5000       # farklı port
"""

import argparse
import sys
from pathlib import Path

# Proje kökünü Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="ThreatSight API Server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Dinlenecek adres (varsayılan: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port numarası (varsayılan: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Geliştirme modu: dosya değişince otomatik yeniden başla")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Uvicorn bulunamadı. Kurmak için:")
        print("  pip install fastapi uvicorn[standard]")
        sys.exit(1)

    print("=" * 55)
    print("  ThreatSight REST API")
    print("=" * 55)
    print(f"  Adres  : http://{args.host}:{args.port}")
    print(f"  Swagger: http://{args.host}:{args.port}/docs")
    print(f"  Stream : http://{args.host}:{args.port}/stream")
    print("=" * 55)

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
