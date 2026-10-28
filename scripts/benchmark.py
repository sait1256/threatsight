"""
ThreatSight - Performance Benchmark
=====================================
Modelin hızını, gecikmeyi ve GPU bellek kullanımını ölçer.

Kullanım:
    python scripts/benchmark.py
    python scripts/benchmark.py --model models/threat_v1_best.pt
    python scripts/benchmark.py --runs 200
"""

import argparse
import time
import numpy as np
from pathlib import Path


def benchmark(args):
    import torch
    from ultralytics import YOLO

    print("=" * 55)
    print("  ThreatSight — Performance Benchmark")
    print("=" * 55)

    model = YOLO(args.model)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print(f"\n  Model  : {args.model}")
    print(f"  Device : {device.upper()}")
    print(f"  Runs   : {args.runs}")
    print(f"  ImgSz  : {args.imgsz}x{args.imgsz}")

    # GPU bilgisi
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU    : {gpu_name} ({total_mem:.1f} GB)")

    # Dummy frame oluştur (gerçek kamera gibi)
    dummy_frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # Isınma turları (ilk frame her zaman yavaştır)
    print("\n[1/3] Isınma turları (3x)...")
    for _ in range(3):
        model(dummy_frame, device=device, verbose=False)

    # GPU belleği ölç
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark döngüsü
    print(f"[2/3] Benchmark ({args.runs} frame)...")
    latencies = []

    for i in range(args.runs):
        start = time.perf_counter()
        _ = model(dummy_frame, device=device, verbose=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms cinsinden

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.runs} tamamlandı...")

    # Sonuçları hesapla
    latencies = np.array(latencies)
    avg_ms   = np.mean(latencies)
    min_ms   = np.min(latencies)
    max_ms   = np.max(latencies)
    p95_ms   = np.percentile(latencies, 95)
    p99_ms   = np.percentile(latencies, 99)
    avg_fps  = 1000.0 / avg_ms

    # GPU bellek kullanımı
    gpu_mem_mb = 0
    if device == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Rapor
    print("\n" + "=" * 55)
    print("  BENCHMARK SONUÇLARI")
    print("=" * 55)
    print(f"  Ortalama gecikme  : {avg_ms:.2f} ms")
    print(f"  Min gecikme       : {min_ms:.2f} ms")
    print(f"  Max gecikme       : {max_ms:.2f} ms")
    print(f"  P95 gecikme       : {p95_ms:.2f} ms")
    print(f"  P99 gecikme       : {p99_ms:.2f} ms")
    print(f"  Ortalama FPS      : {avg_fps:.1f}")
    if gpu_mem_mb > 0:
        print(f"  GPU bellek        : {gpu_mem_mb:.1f} MB")

    # Değerlendirme
    print("\n  Değerlendirme:")
    if avg_fps >= 30:
        print(f"  ✅ Gerçek zamanlı ({avg_fps:.0f} FPS ≥ 30 FPS)")
    elif avg_fps >= 15:
        print(f"  ⚠️  Kabul edilebilir ({avg_fps:.0f} FPS, 15-30 arası)")
    else:
        print(f"  ❌ Yavaş ({avg_fps:.0f} FPS < 15 FPS) — GPU önerilir")

    print("=" * 55)

    # Sonuçları dosyaya kaydet
    if args.save:
        import json
        report = {
            "model": str(args.model),
            "device": device,
            "runs": args.runs,
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "avg_fps": round(avg_fps, 1),
            "gpu_mem_mb": round(gpu_mem_mb, 1),
        }
        out = Path("logs/benchmark_report.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n  Rapor kaydedildi: {out}")


def main():
    parser = argparse.ArgumentParser(description="ThreatSight Benchmark")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Sonuçları JSON'a kaydet")
    main_args = parser.parse_args()
    benchmark(main_args)


if __name__ == "__main__":
    main()
