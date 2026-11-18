"""Atlas de glifos: renderiza cada carácter una vez y ordena por tinta real."""

from dataclasses import dataclass

import numpy as np
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter

CANDIDATES = " .'`^\",:;Il!i><~=+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Paletas de símbolos limpios (se ordenan por tinta medida). None = todos los candidatos.
PRESETS: dict[str, str | None] = {
    "Simple": " .:-=+*#%@",
    "Detallada": " .,:;-~=+!*ox#%&@",
    "Extendida": None,
}


@dataclass(frozen=True)
class Atlas:
    chars: str            # ordenados de menos a más tinta
    bitmaps: np.ndarray   # (N, alto_celda, ancho_celda) uint8
    cell_w: int
    cell_h: int

    @property
    def aspect(self) -> float:
        """ancho/alto de la celda, para calcular la rejilla."""
        return self.cell_w / self.cell_h


def _font(family: str, px: int) -> QFont:
    font = QFont(family)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(px)
    return font


def build_atlas(chars: str = CANDIDATES, family: str = "Menlo", px: int = 16) -> Atlas:
    """Renderiza los glifos (requiere QGuiApplication) y los ordena por tinta."""
    font = _font(family, px)
    fm = QFontMetrics(font)
    cw, ch = fm.horizontalAdvance("M"), fm.height()
    maps = []
    for c in chars:
        img = QImage(cw, ch, QImage.Format.Format_Grayscale8)
        img.fill(0)
        p = QPainter(img)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(0, fm.ascent(), c)
        p.end()
        buf = np.frombuffer(img.constBits(), np.uint8).reshape(ch, img.bytesPerLine())
        maps.append(buf[:, :cw].copy())
    bitmaps = np.stack(maps)
    ink = bitmaps.reshape(len(chars), -1).sum(axis=1)
    order = np.argsort(ink, kind="stable")
    return Atlas("".join(chars[i] for i in order), bitmaps[order], cw, ch)


def pick_palette(atlas: Atlas, chars: str | None) -> Atlas:
    """Sub-atlas con los caracteres pedidos, ordenados por tinta (None = todos)."""
    if chars is None:
        return atlas
    keep = [i for i, c in enumerate(atlas.chars) if c in chars]
    return Atlas("".join(atlas.chars[i] for i in keep), atlas.bitmaps[keep],
                 atlas.cell_w, atlas.cell_h)
