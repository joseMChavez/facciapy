"""Modos de render puros (numpy/cv2, sin Qt): gris de alta resolución -> índices de glifo.

Los modos Braille y Bordes trabajan sobre una rejilla de 2x4 puntos por celda.
"""

import cv2
import numpy as np

DOTS_X, DOTS_Y = 2, 4

# Matriz de Bayer 4x4 para dithering ordenado (estable entre frames, sin ruido propio)
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)

# Peso Unicode de cada punto Braille, por (fila, columna) dentro de la celda
BRAILLE_BITS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], np.uint16)
BRAILLE_BASE = 0x2800

# Orden de las clases de borde: índices levels + k
EDGE_CHARS = "|/-\\"


def _crop(gray_hi: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Recorta/rellena a múltiplos de la celda (2x4 puntos); mínimo una celda."""
    h, w = gray_hi.shape
    rows, cols = max(1, h // DOTS_Y), max(1, w // DOTS_X)
    g = gray_hi[: rows * DOTS_Y, : cols * DOTS_X]
    ph, pw = rows * DOTS_Y - g.shape[0], cols * DOTS_X - g.shape[1]
    if ph or pw:
        g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
    return g, rows, cols


def braille_indices(gray_hi: np.ndarray) -> np.ndarray:
    """(4*rows, 2*cols) gris 0..255 -> (rows, cols) código Braille 0..255 (U+2800 + código)."""
    g, rows, cols = _crop(gray_hi)
    h, w = g.shape
    thr = (np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w] + 0.5) / 16 * 255
    dots = (g > thr).reshape(rows, DOTS_Y, cols, DOTS_X)
    code = (dots * BRAILLE_BITS[None, :, None, :]).sum(axis=(1, 3))
    return code.astype(np.uint8)


def edge_indices(gray_hi: np.ndarray, levels: int, sensitivity: float = 0.5) -> np.ndarray:
    """Índices (rows, cols): sombra 0..levels-1, o levels+k si la celda es un borde.

    k indexa EDGE_CHARS ("|", "/", "-", "\\"). `sensitivity` 0..1: mayor = más bordes.
    """
    g, rows, cols = _crop(gray_hi)
    g = np.ascontiguousarray(g, dtype=np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) / 4  # un escalón de D niveles de gris da mag ≈ D

    def per_cell(a: np.ndarray) -> np.ndarray:  # (rows, cols, 8) puntos de cada celda
        return a.reshape(rows, DOTS_Y, cols, DOTS_X).transpose(0, 2, 1, 3).reshape(rows, cols, -1)

    m = per_cell(mag)
    best = m.argmax(axis=2)[..., None]
    peak = np.take_along_axis(m, best, axis=2)[..., 0]
    bx = np.take_along_axis(per_cell(gx), best, axis=2)[..., 0]
    by = np.take_along_axis(per_cell(gy), best, axis=2)[..., 0]

    # El borde es perpendicular al gradiente. Ángulo del gradiente plegado a [0, π):
    # 0 -> "|", π/4 -> "/", π/2 -> "-", 3π/4 -> "\"
    theta = np.mod(np.arctan2(by, bx), np.pi)
    k = np.rint(theta / (np.pi / 4)).astype(np.int64) % 4

    shade = g.reshape(rows, DOTS_Y, cols, DOTS_X).mean(axis=(1, 3))
    idx = np.clip(np.rint(shade / 255 * (levels - 1)), 0, levels - 1).astype(np.int64)
    min_delta = 5 + 75 * (1 - float(np.clip(sensitivity, 0, 1)))
    return np.where(peak > min_delta, levels + k, idx).astype(np.uint8)
