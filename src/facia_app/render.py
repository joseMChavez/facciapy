"""Índices de nivel -> imagen / texto."""

import cv2
import numpy as np
from PySide6.QtGui import QImage, qRgb

from .glyphs import Atlas

# nombre -> (fondo, tinta)
THEMES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "Mono": ((13, 15, 18), (235, 240, 245)),
    "Fósforo": ((5, 12, 6), (90, 255, 120)),
    "Ámbar": ((16, 10, 2), (255, 176, 0)),
    "Color real": ((0, 0, 0), (255, 255, 255)),
}
COLOR_THEME = "Color real"


def compose(idx: np.ndarray, atlas: Atlas) -> np.ndarray:
    """Compone la imagen de cobertura (H*alto_celda, W*ancho_celda) con indexación numpy."""
    rows, cols = idx.shape
    tiles = atlas.bitmaps[idx]  # (rows, cols, ch, cw)
    return tiles.transpose(0, 2, 1, 3).reshape(rows * atlas.cell_h, cols * atlas.cell_w)


def to_qimage(gray: np.ndarray, theme: str) -> QImage:
    bg, fg = THEMES.get(theme, THEMES["Mono"])
    h, w = gray.shape
    gray = np.ascontiguousarray(gray)
    img = QImage(gray.data, w, h, w, QImage.Format.Format_Indexed8)
    table = [
        qRgb(*(int(bg[k] + (fg[k] - bg[k]) * v / 255) for k in range(3))) for v in range(256)
    ]
    img.setColorTable(table)
    return img.copy()  # desacopla del buffer numpy


def to_qimage_color(gray: np.ndarray, colors: np.ndarray, cell_w: int, cell_h: int) -> QImage:
    """Tiñe cada celda con su color: píxel = color * cobertura (fondo negro).

    Usa operaciones de OpenCV (SIMD) en lugar de aritmética numpy a resolución completa.
    """
    h, w = gray.shape
    big = cv2.resize(np.ascontiguousarray(colors), (w, h), interpolation=cv2.INTER_NEAREST)
    cov = cv2.merge([gray, gray, gray])
    out = np.ascontiguousarray(cv2.multiply(big, cov, scale=1 / 255))
    return QImage(out.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def to_text(idx: np.ndarray, atlas: Atlas) -> str:
    chars = np.frombuffer(atlas.chars.encode("utf-8"), np.uint8)[idx]
    chars = np.hstack([chars, np.full((idx.shape[0], 1), 10, np.uint8)])
    return chars.tobytes().decode("utf-8").rstrip("\n")
