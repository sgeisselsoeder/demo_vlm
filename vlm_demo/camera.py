"""Webcam-Erfassung in einem eigenen Thread. Hält immer nur das neueste Bild vor."""

from __future__ import annotations

import glob
import os
import threading
import time

import cv2
import numpy as np


def list_cameras() -> list[dict]:
    """Findet V4L2-Geräte, die Bilder liefern (Primärknoten, index == 0)."""
    cams = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int(p.removeprefix("/dev/video"))):
        sys_dir = f"/sys/class/video4linux/{os.path.basename(dev)}"
        try:
            with open(f"{sys_dir}/index") as f:
                if f.read().strip() != "0":
                    continue
            with open(f"{sys_dir}/name") as f:
                name = f.read().strip()
        except OSError:
            name = dev
        cams.append({"index": int(dev.removeprefix("/dev/video")), "name": name})
    return cams


class Camera:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width, self.height, self.fps = width, height, fps
        self.index: int | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None  # BGR
        self._frame_id = 0
        self._jpeg: bytes | None = None
        self._jpeg_id = -1
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._running = False
        self._thread: threading.Thread | None = None
        self.mirror = True
        self.error: str | None = None

    def open(self, index: int) -> None:
        self.close()
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error = f"Kamera {index} lässt sich nicht öffnen"
            raise RuntimeError(self.error)
        self.error = None
        self._cap, self.index = cap, index
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self._cap = self._thread = None

    def _loop(self) -> None:
        fails = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    self.error = "Kamera liefert keine Bilder mehr"
                time.sleep(0.02)
                continue
            fails = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._new_frame.notify_all()

    def latest(self) -> np.ndarray | None:
        """Neuestes Bild als RGB-Array (Kopie)."""
        with self._lock:
            if self._frame is None:
                return None
            return cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)

    def mjpeg(self, quality: int = 80):
        """Generator für einen multipart/x-mixed-replace-Stream."""
        last_id = -1
        while True:
            with self._new_frame:
                if not self._new_frame.wait_for(lambda: self._frame_id != last_id, timeout=1.0):
                    continue
                last_id = self._frame_id
                # JPEG nur einmal pro Bild kodieren, auch bei mehreren Clients
                if self._jpeg_id != last_id:
                    ok, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not ok:
                        continue
                    self._jpeg, self._jpeg_id = buf.tobytes(), last_id
                jpeg = self._jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
