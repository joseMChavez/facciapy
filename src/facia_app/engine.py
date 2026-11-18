"""Procesamiento puro: frame BGR -> matriz de índices de paleta. Sin Qt ni cámara."""

import cv2
import numpy as np

COLOR_FLOOR = 40.0   # por debajo, el color no se normaliza (evita ruido en zonas oscuras)
SATURATION = 1.25
WORK_WIDTH = 320  # resolución de trabajo para CLAHE (menos ruido que sobre la miniatura)


class Engine:
    """Mantiene estado temporal (EMA) entre frames."""

    def __init__(self, levels: int, gamma: float = 0.8, smoothing: float = 0.4):
        self.levels = levels
        self.smoothing = smoothing
        self._gamma = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._ema: np.ndarray | None = None
        self._color_ema: np.ndarray | None = None

    def process(self, frame: np.ndarray, cols: int, rows: int) -> np.ndarray:
        """Devuelve índices uint8 de forma (rows, cols). Tolera tamaños de 0 o 1."""
        cols, rows = max(1, cols), max(1, rows)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        work = cv2.resize(gray, (WORK_WIDTH, max(1, WORK_WIDTH * h // w)),
                          interpolation=cv2.INTER_AREA)
        work = self._clahe.apply(work)
        work = cv2.LUT(work, self._gamma)
        small = cv2.resize(work, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)

        if self._ema is None or self._ema.shape != small.shape:
            self._ema = small
        else:
            self._ema = self.smoothing * small + (1 - self.smoothing) * self._ema

        idx = np.rint(self._ema / 255 * (self.levels - 1))
        return np.clip(idx, 0, self.levels - 1).astype(np.uint8)


    def colors(self, frame: np.ndarray, cols: int, rows: int) -> np.ndarray:
        """Color por celda (rows, cols, 3) RGB uint8, vivo y estable.

        El brillo ya lo aporta la cobertura del glifo, así que aquí se conserva el
        matiz: se normaliza el canal máximo y se refuerza un poco la saturación.
        """
        cols, rows = max(1, cols), max(1, rows)
        h, w = frame.shape[:2]
        # Dos pasos: primero a la resolución de trabajo (barato), luego a la rejilla
        work = cv2.resize(frame, (WORK_WIDTH, max(1, WORK_WIDTH * h // w)),
                          interpolation=cv2.INTER_AREA)
        small = cv2.resize(work, (cols, rows), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self._color_ema is None or self._color_ema.shape != rgb.shape:
            self._color_ema = rgb
        else:
            self._color_ema = self.smoothing * rgb + (1 - self.smoothing) * self._color_ema
        c = self._color_ema

        peak = np.maximum(c.max(axis=2, keepdims=True), COLOR_FLOOR)  # no amplificar el ruido
        c = c / peak * 255
        c = c.mean(axis=2, keepdims=True) + (c - c.mean(axis=2, keepdims=True)) * SATURATION
        return np.clip(c, 0, 255).astype(np.uint8)


def fit_grid(frame_shape: tuple[int, int], max_cols: int, max_rows: int,
             char_aspect: float = 0.5) -> tuple[int, int]:
    """Calcula (cols, rows) que caben en el área conservando la proporción."""
    h, w = frame_shape
    max_cols, max_rows = max(1, max_cols), max(1, max_rows)
    cols = max_cols
    rows = max(1, int(cols * h / w * char_aspect))
    if rows > max_rows:
        rows = max_rows
        cols = max(1, int(rows * w / h / char_aspect))
    return cols, rows
