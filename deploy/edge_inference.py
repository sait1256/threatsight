"""
ThreatSight — Edge Inference Engine
Raspberry Pi / Jetson Nano için optimize edilmiş çıkarım motoru.

Özellikler:
- ONNX Runtime ile düşük gecikmeli çıkarım
- GPIO LED/buzzer entegrasyonu (opsiyonel)
- MQTT ile merkezi sunucuya alert gönderimi
- Headless mod (display olmadan çalışır)
- Düşük bellek kullanımı için single-threaded pipeline
"""

import argparse
import json
import time
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np

# ─── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/edge_inference.log"),
    ]
)
log = logging.getLogger("edge")

# ─── Sabitler ─────────────────────────────────────────────────────────────────
CLASSES = {0: "knife", 1: "pistol", 2: "drone"}
THREAT_LEVEL = {"knife": 2, "pistol": 3, "drone": 2}
COLORS = {1: (0, 200, 0), 2: (0, 165, 255), 3: (0, 0, 220)}

# ─── ONNX Runtime yükle ───────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    log.warning("onnxruntime bulunamadı — YOLOv8 native kullanılacak")

# ─── MQTT (opsiyonel) ─────────────────────────────────────────────────────────
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    log.info("paho-mqtt bulunamadı — MQTT devre dışı")

# ─── GPIO (opsiyonel, sadece RPi) ─────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False


class GPIOController:
    """Raspberry Pi GPIO üzerinden LED ve buzzer kontrolü."""

    LED_PIN = 17
    BUZZER_PIN = 27

    def __init__(self):
        if not GPIO_AVAILABLE:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LED_PIN, GPIO.OUT)
        GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
        log.info("GPIO başlatıldı (LED=%d, BUZZER=%d)", self.LED_PIN, self.BUZZER_PIN)

    def alert(self, threat_level: int):
        """Tehdit seviyesine göre uyarı ver."""
        if not GPIO_AVAILABLE:
            return
        if threat_level >= 3:
            GPIO.output(self.LED_PIN, GPIO.HIGH)
            GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(self.BUZZER_PIN, GPIO.LOW)
        elif threat_level == 2:
            GPIO.output(self.LED_PIN, GPIO.HIGH)

    def reset(self):
        if not GPIO_AVAILABLE:
            return
        GPIO.output(self.LED_PIN, GPIO.LOW)
        GPIO.output(self.BUZZER_PIN, GPIO.LOW)

    def cleanup(self):
        if GPIO_AVAILABLE:
            GPIO.cleanup()


class MQTTPublisher:
    """Tehdit alertlerini MQTT broker'a gönderir."""

    def __init__(self, broker: str, port: int = 1883, topic: str = "threatsight/alerts"):
        self.topic = topic
        self.client = None
        if not MQTT_AVAILABLE:
            log.info("MQTT devre dışı")
            return
        try:
            self.client = mqtt.Client()
            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()
            log.info("MQTT bağlandı: %s:%d/%s", broker, port, topic)
        except Exception as e:
            log.warning("MQTT bağlantı hatası: %s", e)
            self.client = None

    def publish(self, payload: dict):
        if self.client is None:
            return
        try:
            self.client.publish(self.topic, json.dumps(payload))
        except Exception as e:
            log.warning("MQTT gönderim hatası: %s", e)

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


class ONNXDetector:
    """ONNX Runtime tabanlı YOLOv8 çıkarım motoru."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, input_size: int = 640):
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime kurulu değil: pip install onnxruntime")

        # CPU veya GPU provider seç
        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            log.info("CUDA (GPU) kullanılıyor")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        log.info("ONNX model yüklendi: %s", model_path)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Frame'i ONNX girişi için hazırla."""
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def postprocess(
        self, outputs: np.ndarray, orig_shape: tuple
    ) -> List[Dict]:
        """Model çıktısını Detection listesine dönüştür."""
        detections = []
        predictions = outputs[0][0]  # (num_boxes, 4+nc)

        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for pred in predictions.T:
            x_c, y_c, w, h = pred[:4]
            class_scores = pred[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.conf_threshold:
                continue

            x1 = int((x_c - w / 2) * scale_x)
            y1 = int((y_c - h / 2) * scale_y)
            x2 = int((x_c + w / 2) * scale_x)
            y2 = int((y_c + h / 2) * scale_y)

            detections.append({
                "class_id": cls_id,
                "label": CLASSES.get(cls_id, "unknown"),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

        return detections

    def detect(self, frame: np.ndarray) -> List[Dict]:
        inp = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.postprocess(outputs, frame.shape)


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Tespitleri frame üzerine çiz."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        threat = THREAT_LEVEL.get(label, 1)
        color = COLORS.get(threat, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def run(args):
    """Ana çıkarım döngüsü."""
    os.makedirs("logs", exist_ok=True)

    # Detector
    detector = ONNXDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        input_size=args.imgsz,
    )

    # GPIO
    gpio = GPIOController()

    # MQTT
    mqtt_pub = MQTTPublisher(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=args.mqtt_topic,
    ) if args.mqtt_broker else None

    # Kamera
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("Kamera açılamadı: %s", args.source)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    log.info("Edge çıkarım başladı | kaynak=%s | model=%s", args.source, args.model)

    alert_cooldown: Dict[str, float] = {}
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame alınamadı, yeniden deneniyor...")
                time.sleep(0.1)
                continue

            frame_count += 1
            detections = detector.detect(frame)

            # En yüksek tehdit seviyesi
            max_threat = 0
            for det in detections:
                threat = THREAT_LEVEL.get(det["label"], 1)
                if threat > max_threat:
                    max_threat = threat

            # GPIO uyarısı
            if max_threat >= 2:
                gpio.alert(max_threat)
            else:
                gpio.reset()

            # Alert gönder (cooldown ile)
            now = time.time()
            for det in detections:
                label = det["label"]
                threat = THREAT_LEVEL.get(label, 1)
                if threat < 2:
                    continue
                last_alert = alert_cooldown.get(label, 0)
                if now - last_alert >= args.cooldown:
                    alert_cooldown[label] = now
                    payload = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "label": label,
                        "confidence": round(det["confidence"], 3),
                        "threat_level": threat,
                        "device_id": args.device_id,
                    }
                    log.warning("⚠ ALERT: %s", json.dumps(payload))
                    if mqtt_pub:
                        mqtt_pub.publish(payload)

            # Görüntü kaydet (display varsa)
            if not args.headless:
                annotated = draw_detections(frame.copy(), detections)
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 1e-9)
                cv2.putText(annotated, f"FPS: {fps:.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("ThreatSight Edge", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Kullanıcı tarafından durduruldu")
    finally:
        cap.release()
        gpio.cleanup()
        if mqtt_pub:
            mqtt_pub.disconnect()
        if not args.headless:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps = frame_count / max(elapsed, 1e-9)
        log.info("Oturum bitti | %d frame | %.1f FPS | %.1f sn", frame_count, fps, elapsed)


def parse_args():
    p = argparse.ArgumentParser(description="ThreatSight Edge Inference")
    p.add_argument("--model", default="models/threat_v1_best.onnx",
                   help="ONNX model dosyası")
    p.add_argument("--source", default=0, help="Kamera index veya video dosyası")
    p.add_argument("--conf", type=float, default=0.5, help="Güven eşiği")
    p.add_argument("--imgsz", type=int, default=640, help="Giriş boyutu")
    p.add_argument("--width", type=int, default=640, help="Kamera genişliği")
    p.add_argument("--height", type=int, default=480, help="Kamera yüksekliği")
    p.add_argument("--cooldown", type=float, default=5.0, help="Alert cooldown (sn)")
    p.add_argument("--headless", action="store_true", help="Display olmadan çalış")
    p.add_argument("--mqtt-broker", default="", help="MQTT broker adresi")
    p.add_argument("--mqtt-port", type=int, default=1883, help="MQTT port")
    p.add_argument("--mqtt-topic", default="threatsight/alerts", help="MQTT topic")
    p.add_argument("--device-id", default="edge-01", help="Cihaz kimliği")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
