# 🎯 ThreatSight

**Real-time AI-powered threat detection and classification system**

> Built for defense & security AI portfolio — GPU-accelerated, production-ready architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Real-time inference** — YOLOv8 on GPU, 30+ FPS on mid-range hardware
- **Multi-class threat scoring** — Objects classified as LOW / MEDIUM / HIGH threat
- **Alert system** — Triggers on high-threat detections with configurable cooldown
- **Event logging** — SQLite database with timestamped records and snapshots
- **Thread-safe video pipeline** — Non-blocking frame capture, supports webcam / video file / RTSP
- **Headless mode** — Run without display for server deployment

## Architecture

```
threatsight/
├── main.py                  # Entry point
├── config/config.yaml       # All settings in one place
└── src/
    ├── detector.py          # YOLOv8 inference engine
    ├── video_stream.py      # Threaded camera pipeline
    ├── threat_scorer.py     # Threat level evaluation + alerts
    ├── logger.py            # SQLite event logging
    └── visualizer.py        # OpenCV overlay renderer
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> CUDA recommended. CPU fallback available with `--cpu` flag.

### 2. Run

```bash
# Webcam
python main.py

# Video file
python main.py --source path/to/video.mp4

# Adjust confidence
python main.py --conf 0.5

# CPU mode (no GPU)
python main.py --cpu

# Headless / server mode
python main.py --no-display
```

Press **Q** or **ESC** to quit.

### 3. Configuration

Edit `config/config.yaml` to customize:
- Model weights (`yolov8n` → `yolov8m` for higher accuracy)
- Threat levels per object class
- Alert thresholds and cooldown
- Logging and snapshot settings

## Threat Levels

| Level | Label  | Example Classes         | Color  |
|-------|--------|--------------------------|--------|
| 1     | LOW    | person, car, backpack    | 🟢 Green  |
| 2     | MEDIUM | truck, airplane          | 🟠 Orange |
| 3     | HIGH   | gun, knife, drone, helicopter | 🔴 Red |

Customize classes and levels freely in `config.yaml`.

## Roadmap

- [x] Phase 1: Core detection pipeline
- [ ] Phase 2: PyQt6 professional UI
- [ ] Phase 3: Custom model training (domain-specific threats)
- [ ] Phase 4: MLOps — model versioning, CI/CD, monitoring
- [ ] Phase 5: Edge deployment (Raspberry Pi / Jetson Nano)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8 (Ultralytics) |
| Inference | PyTorch + CUDA |
| Video I/O | OpenCV |
| Logging   | SQLite |
| Config    | YAML |