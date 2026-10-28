"""
ThreatSight — Real-time AI Threat Detection System
====================================================
Kullanım:
    python main.py                     # Webcam
    python main.py --source video.mp4  # Video dosyası
    python main.py --conf 0.5          # Confidence ayarı
    python main.py --cpu               # GPU yoksa

Çıkış: Q tuşu veya Ctrl+C
"""

import cv2
import yaml
import argparse
import time

from src.detector import ThreatDetector
from src.video_stream import VideoStream
from src.threat_scorer import ThreatScorer
from src.logger import EventLogger
from src.visualizer import Visualizer


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="ThreatSight")
    parser.add_argument("--source", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.source is not None:
        try:
            config["camera"]["source"] = int(args.source)
        except ValueError:
            config["camera"]["source"] = args.source

    if args.conf is not None:
        config["model"]["confidence"] = args.conf

    if args.cpu:
        config["model"]["device"] = "cpu"

    print("=" * 50)
    print("  ThreatSight — Real-time Threat Detection")
    print("=" * 50)

    detector = ThreatDetector(config)
    detector.warmup()

    scorer  = ThreatScorer(config)
    logger  = EventLogger(config)
    visualizer = Visualizer(config)

    cam_cfg = config["camera"]
    stream = VideoStream(
        source=cam_cfg["source"],
        resolution=tuple(cam_cfg["resolution"]),
        fps_limit=cam_cfg["fps_limit"],
    )

    active_alert = None
    alert_shown_at = 0.0
    ALERT_DURATION = 3.0

    try:
        stream.start()
        print("\n[BAŞLADI] Çıkmak için Q tuşuna bas.\n")

        while True:
            ret, frame = stream.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Tespit
            result = detector.detect(frame)

            # Tehdit değerlendirme
            alert = scorer.evaluate(result)
            if alert:
                print(f"[ALERT] {alert.message}")
                logger.log_alert(alert, frame)
                active_alert = alert
                alert_shown_at = time.time()

            # Alert süresi bittiyse temizle
            if active_alert and (time.time() - alert_shown_at) > ALERT_DURATION:
                active_alert = None

            # Görüntü
            if not args.no_display:
                annotated = visualizer.draw(frame, result, scorer, active_alert)
                cv2.imshow("ThreatSight", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\n[!] Durduruldu.")

    finally:
        stream.stop()
        logger.close()
        cv2.destroyAllWindows()

        stats = logger.get_stats()
        print("\n" + "=" * 50)
        print("  OTURUM ÖZETİ")
        print("=" * 50)
        print(f"  Toplam alert  : {stats.get('total_alerts', 0)}")
        print(f"  Yüksek tehdit : {stats.get('high_threat_alerts', 0)}")
        print(f"  Çalışma süresi: {stream.uptime:.1f} sn")
        print(f"  Frame sayısı  : {stream.frame_count}")
        print("=" * 50)


if __name__ == "__main__":
    main()