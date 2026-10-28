"""
ThreatSight - Model Evaluation
================================
Eğitilmiş custom modeli test eder ve metrik hesaplar.

Kullanım:
    # Custom modeli değerlendir
    python scripts/evaluate.py --model models/threat_v1_best.pt

    # Belirli bir görüntü veya klasörde test
    python scripts/evaluate.py --model models/threat_v1_best.pt --source data/test_images/
"""

import argparse
from pathlib import Path


def evaluate(args):
    from ultralytics import YOLO
    import cv2

    print("=" * 55)
    print("  ThreatSight — Model Evaluation")
    print("=" * 55)

    if not Path(args.model).exists():
        print(f"[HATA] Model bulunamadı: {args.model}")
        print("Önce şunu çalıştır: python scripts/train.py")
        return

    model = YOLO(args.model)

    print(f"\n  Model  : {args.model}")
    print(f"  Kaynak : {args.source}")
    print(f"  Conf   : {args.conf}")

    if args.val:
        # Validation seti üzerinde resmi metrikler
        print("\n[VAL] Validation metrikleri hesaplanıyor...")
        metrics = model.val(
            data=args.data,
            conf=args.conf,
            iou=0.6,
            plots=True,
            save_json=True,
        )

        print("\n📊 SONUÇLAR:")
        print(f"  Precision : {metrics.results_dict.get('metrics/precision(B)', 0):.3f}")
        print(f"  Recall    : {metrics.results_dict.get('metrics/recall(B)', 0):.3f}")
        print(f"  mAP50     : {metrics.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"  mAP50-95  : {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")

    else:
        # Kaynak üzerinde tahmin
        print(f"\n[PREDICT] Tahmin yapılıyor: {args.source}")
        results = model.predict(
            source=args.source,
            conf=args.conf,
            save=True,
            save_txt=True,
            project="runs/eval",
            name="threat_eval",
        )

        print(f"\n✅ Sonuçlar kaydedildi: runs/eval/threat_eval/")
        print(f"   Toplam tespit: {sum(len(r.boxes) for r in results)}")

        # Sınıf bazlı özet
        class_counts = {}
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                name = model.names[cls]
                class_counts[name] = class_counts.get(name, 0) + 1

        if class_counts:
            print("\n  Tespit özeti:")
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {cls:10s}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="ThreatSight Model Evaluation")
    parser.add_argument("--model", default="models/threat_v1_best.pt",
                        help="Değerlendirilecek model .pt dosyası")
    parser.add_argument("--source", default="data/threat_dataset/test/images",
                        help="Test görüntüleri (klasör, dosya veya 0 webcam)")
    parser.add_argument("--data", default="data/dataset.yaml",
                        help="Dataset YAML (--val için)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--val", action="store_true",
                        help="Resmi validation metrikleri hesapla")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()