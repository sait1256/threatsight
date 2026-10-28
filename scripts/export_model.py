"""
ThreatSight - Model Export
============================
Eğitilmiş modeli farklı formatlara dönüştürür:
  - ONNX     → Platform bağımsız (Windows/Linux/Mac)
  - TensorRT → NVIDIA GPU'da maksimum hız (Jetson/RTX)
  - CoreML   → Apple Silicon (Mac deployment)

Kullanım:
    python scripts/export_model.py
    python scripts/export_model.py --model models/threat_v1_best.pt --format onnx
    python scripts/export_model.py --format tensorrt  # En hızlı, sadece NVIDIA GPU
"""

import argparse
from pathlib import Path


SUPPORTED_FORMATS = ["onnx", "tensorrt", "coreml", "openvino"]


def export(args):
    from ultralytics import YOLO

    print("=" * 55)
    print("  ThreatSight — Model Export")
    print("=" * 55)
    print(f"\n  Model  : {args.model}")
    print(f"  Format : {args.format.upper()}")
    print(f"  ImgSz  : {args.imgsz}")

    if not Path(args.model).exists():
        print(f"\n[HATA] Model bulunamadı: {args.model}")
        print("Önce: python scripts/train.py")
        return

    model = YOLO(args.model)

    print(f"\n[Export] {args.format.upper()} formatına dönüştürülüyor...")

    # Format'a göre export seçenekleri
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "simplify": True,    # ONNX graph sadeleştir
        "dynamic": False,    # Sabit batch size (deployment için daha iyi)
    }

    if args.format == "tensorrt":
        export_kwargs["half"] = True     # FP16 — 2x hız artışı
        export_kwargs["workspace"] = 4   # GB cinsinden TRT workspace

    if args.format == "onnx":
        export_kwargs["opset"] = 17      # ONNX opset versiyonu

    exported_path = model.export(**export_kwargs)

    print("\n✅ Export tamamlandı!")
    print(f"   Konum : {exported_path}")

    # Dosya boyutu
    out = Path(str(exported_path))
    if out.exists():
        size_mb = out.stat().st_size / 1024**2
        print(f"   Boyut : {size_mb:.1f} MB")

    # Kullanım talimatı
    print("\n  Bu modeli kullanmak için:")
    if args.format == "onnx":
        print("    from ultralytics import YOLO")
        print(f"    model = YOLO('{exported_path}')")
        print("    # ONNX modeli CPU'da da hızlı çalışır!")
    elif args.format == "tensorrt":
        print("    from ultralytics import YOLO")
        print(f"    model = YOLO('{exported_path}')")
        print("    # TensorRT: En yüksek GPU performansı")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="ThreatSight Model Export")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Export edilecek model (.pt)")
    parser.add_argument("--format", default="onnx",
                        choices=SUPPORTED_FORMATS,
                        help=f"Export formatı: {', '.join(SUPPORTED_FORMATS)}")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Model giriş boyutu")
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
