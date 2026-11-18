"""Seguimiento de rostro: detecta con YuNet y recorta de forma suave. Sin Qt.

El modelo NO se incluye ni se descarga solo: ver README (models/).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             + MODEL_NAME)
ENV_VAR = "FACIA_FACE_MODEL"

DET_WIDTH = 320    # resolución de detección: suficiente y barata
MIN_CROP = 0.35    # el recorte nunca baja de esta fracción del alto (no acercarse en exceso)


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0


class Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Face]: ...


def model_path() -> Path | None:
    """Ruta del modelo si existe: variable FACIA_FACE_MODEL o models/ del proyecto."""
    candidates = [Path(p) for p in (os.environ.get(ENV_VAR),) if p]
    candidates.append(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
    return next((p for p in candidates if p.is_file()), None)


class YuNetDetector:
    """Detector YuNet de OpenCV (cv2.FaceDetectorYN). OpenCV 5 ya no trae cascadas Haar."""

    def __init__(self, path: str | Path, score: float = 0.7):
        self._net = cv2.FaceDetectorYN.create(str(path), "", (DET_WIDTH, DET_WIDTH), score, 0.3,
                                              5000)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        self._net.setInputSize((w, h))
        _, faces = self._net.detect(bgr)
        if faces is None:
            return []
        return [Face(*map(float, f[:4]), float(f[14])) for f in faces]


def make_tracker() -> "FaceTracker | None":
    """Tracker con YuNet, o None si falta el modelo. Lanza cv2.error si no se puede cargar."""
    path = model_path()
    return FaceTracker(YuNetDetector(path)) if path else None


class FaceTracker:
    """Recorta el frame alrededor del rostro con movimiento suave.

    - detecta cada `every` frames (a 320 px); el resto solo suaviza el movimiento
    - el recorte conserva la proporción del frame, así la rejilla ASCII no cambia de forma
    - la primera detección fija el encuadre de golpe; después se acerca con EMA
    - sin rostro durante `lost_after` detecciones, vuelve poco a poco al frame completo
    """

    def __init__(self, detector: Detector, every: int = 3, margin: float = 0.4,
                 smoothing: float = 0.25, lost_after: int = 5):
        self.detector = detector
        self.every, self.margin = max(1, every), margin
        self.smoothing, self.lost_after = smoothing, lost_after
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._lost = 0
        self._state: np.ndarray | None = None   # (cx, cy, alto_recorte)
        self._target: np.ndarray | None = None
        self.has_face = False

    def crop(self, frame: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        if self._n % self.every == 0:
            self._detect(frame)
        self._n += 1
        if self._state is None:
            return frame  # aún no se ha visto ningún rostro

        self._state += self.smoothing * (self._target - self._state)
        cx, cy, th = self._state
        th = float(np.clip(th, min(H, 16), H))
        tw = th * W / H
        h, w = max(1, round(th)), min(W, max(1, round(tw)))
        x0 = int(np.clip(round(cx - w / 2), 0, W - w))
        y0 = int(np.clip(round(cy - h / 2), 0, H - h))
        return frame[y0 : y0 + h, x0 : x0 + w]

    def _detect(self, frame: np.ndarray) -> None:
        H, W = frame.shape[:2]
        dw = DET_WIDTH
        small = cv2.resize(frame, (dw, max(1, dw * H // W)), interpolation=cv2.INTER_AREA)
        faces = self.detector.detect(small)
        if faces:
            f = max(faces, key=lambda f: f.w * f.h)
            k = W / dw
            size = max(f.w, f.h) * k * (1 + 2 * self.margin)
            th = float(np.clip(size, H * MIN_CROP, H))
            self._target = np.array([(f.x + f.w / 2) * k, (f.y + f.h / 2) * k, th])
            self._lost, self.has_face = 0, True
            if self._state is None:
                self._state = self._target.copy()
        else:
            self._lost += 1
            self.has_face = False
            if self._state is not None and self._lost >= self.lost_after:
                self._target = np.array([W / 2, H / 2, float(H)])
