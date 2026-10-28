"""
ThreatSight - Custom Model Training
=====================================
YOLOv8'i kendi tehdit tespiti dataseti üzerinde fine-tune eder.

Kullanım:
    # Tam eğitim
    python scripts/train.py

    # Hızlı test (az epoch)
    python scripts/train.py --epochs 5 --quick

    # GPU ile
    python scripts/train.py --device 0

    # CPU ile (yavaş)
    python scripts/train.py --device cpu
"""

import argparse
import yaml
import time
import os
from pathlib import Path


def check_dataset(dataset_yaml: str) -> bool:
    """Dataset hazır mı kontrol et."""
    with open(dataset_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = Path(cfg.get("path", ""))
    train_dir = dataset_path / "train" / "images"

    if not train_dir.exists():
        print(f"[HATA] Train klasörü bulunamadı: {train_dir}")
        print("Önce şunu çalıştır: python scripts/download_dataset.py")
        return False

    image_files = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    print(f"[Dataset] Train görüntüsü: {len(image_files)}")

    if len(image_files) == 0:
        print("[UYARI] Train klasörü boş! Demo moddasın.")
        print("  → Roboflow'dan gerçek dataset indir veya kendi görüntülerini ekle.")
        return False

    return True


def train(args):
    """YOLOv8 fine-tuning."""
    from ultralytics import YOLO

    print("=" * 55)
    print("  ThreatSight — Custom Model Training")
    print("=" * 55)
    print(f"\n  Model     : {args.model}")
    print(f"  Dataset   : {args.data}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Device    : {args.device}")
    print(f"  Proje     : {args.project}/{args.name}")
    print("=" * 55)

    # Dataset kontrolü
    if not Path(args.data).exists():
        print(f"[HATA] Dataset YAML bulunamadı: {args.data}")
        return

    # Model yükle (pretrained COCO ağırlıkları ile başla)
    print(f"\n[1/3] Model yükleniyor: {args.model}")
    model = YOLO(args.model)

    # Fine-tuning başlat
    print(f"[2/3] Eğitim başlıyor... ({args.epochs} epoch)")
    start = time.time()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        
        # Optimizasyon
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Augmentation (veri çoğaltma — küçük dataset için kritik)
        augment=True,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,        # 4 görüntüyü birleştir
        mixup=0.1,         # Görüntü karıştırma
        hsv_h=0.015,       # Renk tonu değişimi
        hsv_s=0.7,         # Doygunluk değişimi
        hsv_v=0.4,         # Parlaklık değişimi
        scale=0.5,         # Ölçek değişimi
        translate=0.1,     # Konum değişimi
        
        # Kaydetme
        save=True,
        save_period=10,    # Her 10 epoch'ta checkpoint
        patience=20,       # Early stopping (20 epoch gelişme yoksa dur)
        
        # Log
        verbose=True,
        plots=True,        # Eğitim grafikleri kaydet
        val=True,          # Her epoch validation
    )

    elapsed = time.time() - start

    # Sonuç
    print("\n" + "=" * 55)
    print("  EĞİTİM TAMAMLANDI")
    print("=" * 55)
    print(f"  Süre      : {elapsed/60:.1f} dakika")
    print(f"  mAP50     : {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"  mAP50-95  : {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
    
    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n  En iyi model: {best_model}")
    print("=" * 55)

    # [3/3] Modeli models/ klasörüne kopyala
    if best_model.exists():
        import shutil
        dest = Path("models") / "threat_v1_best.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(best_model, dest)
        print(f"\n[3/3] Model kopyalandı: {dest}")
        print("\nKullanmak için config/config.yaml'da şunu değiştir:")
        print("  model_path: models/threat_v1_best.pt")

    return results


def main():
    parser = argparse.ArgumentParser(description="ThreatSight Custom Model Training")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Başlangıç modeli (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", default="data/dataset.yaml",
                        help="Dataset YAML dosyası")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Eğitim epoch sayısı")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (GPU belleğine göre ayarla)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Görüntü boyutu")
    parser.add_argument("--device", default="0",
                        help="Cihaz: 0 (GPU), cpu")
    parser.add_argument("--project", default="runs/train",
                        help="Sonuçların kaydedileceği klasör")
    parser.add_argument("--name", default="threat_v1",
                        help="Deney ismi")
    parser.add_argument("--quick", action="store_true",
                        help="Hızlı test: 5 epoch, batch 8")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.batch = 8
        print("[QUICK MOD] 5 epoch, batch 8\n")

    train(args)


if __name__ == "__main__":
    main()