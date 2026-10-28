"""
ThreatSight - Video Stream Manager
Kamera, video dosyası veya RTSP stream'i yönetir.
Thread-safe, non-blocking frame okuma.
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple


class VideoStream:
    def __init__(self, source, resolution: Tuple[int, int] = (1280, 720), fps_limit: int = 30):
        self.source = source
        self.resolution = resolution
        self.fps_limit = fps_limit
        self._frame_delay = 1.0 / fps_limit

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._start_time = 0.0

    def start(self) -> "VideoStream":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Video kaynağı açılamadı: {self.source}")

        w, h = self.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("İlk frame okunamadı.")
        self._frame = frame
        self._running = True
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        print(f"[VideoStream] Başlatıldı: {self.source} — {w}x{h} @ {self.fps_limit}fps")
        return self

    def _reader_loop(self):
        while self._running:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._running = False
                    break

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            elapsed = time.perf_counter() - t0
            sleep_time = self._frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        print("[VideoStream] Durduruldu.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
