"""Captura de cámara en un hilo, tolerante a fallos. Solo guarda el último frame."""

import enum
import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

WARMUP_FRAMES = 5
MAX_CONSECUTIVE_FAILS = 30


class State(enum.Enum):
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class CameraSource:
    def __init__(self, source: int | str = 0):
        self._source = source
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self.state = State.CONNECTING
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> tuple[int, np.ndarray | None]:
        """Devuelve (seq, frame). Si seq no cambió, no hay frame nuevo."""
        with self._lock:
            return self._seq, self._frame

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self.state = State.FAILED
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff, fails, good = 0.5, 0, 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails >= MAX_CONSECUTIVE_FAILS:
                        self.state = State.RECONNECTING
                        break
                    time.sleep(0.03)
                    continue
                fails, good = 0, good + 1
                if good <= WARMUP_FRAMES:
                    continue
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
                self.state = State.LIVE
            cap.release()  # liberado por el mismo hilo que lee
