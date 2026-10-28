"""
ThreatSight - Dataset Downloader
=================================
Roboflow üzerindeki açık kaynak tehdit tespiti veri setlerini indirir.
Kullanım:
    python scripts/download_dataset.py

Not: İnternet bağlantısı ve roboflow paketi gereklidir.
    pip install roboflow
"""

import os
import sys
import argparse
from pathlib import Path


def download_from_roboflow(api_key: str, output_dir: str = "data/threat_dataset"):
    """
    Roboflow'dan knife/pistol/drone dataset indir.
    
    Roboflow Universe'den ücretsiz kullanılabilir dataset'ler:
    - Knife Detection: roboflow.com/universe/projects/knife-detection
    - Pistol Detection: roboflow.com/universe/projects/pistol-detection  
    - Drone Detection: roboflow.com/universe/projects/drone-detection
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[HATA] roboflow paketi bulunamadı.")
        print("Çözüm: pip install roboflow")
        sys.exit(1)

    print("=" * 55)
    print("  ThreatSight — Dataset Downloader")
    print("=" * 55)
    
    os.makedirs(output_dir, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    
    # Knife Detection Dataset (public)
    print("\n[1/3] Knife dataset indiriliyor...")
    project = rf.workspace("roboflow-universe-projects").project("knife-detection-lvn3t")
    version = project.version(3)
    dataset = version.download("yolov8", location=f"{output_dir}/knife")
    print(f"  → {dataset.location}")

    # Pistol Detection Dataset (public)
    print("\n[2/3] Pistol dataset indiriliyor...")
    project = rf.workspace("roboflow-universe-projects").project("pistol-detection-0ne3h")
    version = project.version(5)
    dataset = version.download("yolov8", location=f"{output_dir}/pistol")
    print(f"  → {dataset.location}")

    # Drone Detection Dataset (public)
    print("\n[3/3] Drone dataset indiriliyor...")
    project = rf.workspace("roboflow-universe-projects").project("drone-detection-ioacm")
    version = project.version(1)
    dataset = version.download("yolov8", location=f"{output_dir}/drone")
    print(f"  → {dataset.location}")

    print("\n✅ Tüm datasetler indirildi!")
    print(f"📁 Konum: {output_dir}/")
    print("\nSonraki adım: python scripts/merge_datasets.py")


def download_demo_mode(output_dir: str = "data/threat_dataset"):
    """
    Roboflow API key yoksa demo dataset yapısı oluşturur.
    Kendi görüntülerinle doldurmak için kullanın.
    """
    print("\n[DEMO MOD] Boş dataset yapısı oluşturuluyor...")

    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            path = Path(output_dir) / split / sub
            path.mkdir(parents=True, exist_ok=True)

    # Demo README
    readme = Path(output_dir) / "README.txt"
    readme.write_text(
        "ThreatSight Custom Dataset\n"
        "==========================\n\n"
        "Klasör yapısı:\n"
        "  train/images/  → Eğitim görüntüleri (.jpg/.png)\n"
        "  train/labels/  → YOLOv8 format etiketler (.txt)\n"
        "  val/images/    → Doğrulama görüntüleri\n"
        "  val/labels/    → Doğrulama etiketleri\n\n"
        "Etiket formatı (her satır bir nesne):\n"
        "  <class_id> <x_center> <y_center> <width> <height>\n"
        "  (Tüm değerler 0-1 arasında normalize edilmiş)\n\n"
        "Sınıflar:\n"
        "  0 = knife\n"
        "  1 = pistol\n"
        "  2 = drone\n\n"
        "Roboflow API Key alarak gerçek dataset indirmek için:\n"
        "  python scripts/download_dataset.py --api-key YOUR_KEY\n",
        encoding="utf-8"
    )

    print(f"✅ Demo yapısı oluşturuldu: {output_dir}/")
    print("📄 Detaylar için:", str(readme))


def main():
    parser = argparse.ArgumentParser(description="ThreatSight Dataset Downloader")
    parser.add_argument(
        "--api-key", 
        default=None,
        help="Roboflow API key (roboflow.com/settings/api adresinden al)"
    )
    parser.add_argument(
        "--output", 
        default="data/threat_dataset",
        help="Dataset indirme klasörü"
    )
    args = parser.parse_args()

    if args.api_key:
        download_from_roboflow(args.api_key, args.output)
    else:
        print("ℹ️  API key belirtilmedi → Demo mod")
        print("   Roboflow API key için: https://roboflow.com/settings/api\n")
        download_demo_mode(args.output)


if __name__ == "__main__":
    main()