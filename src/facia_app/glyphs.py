"""Atlas de glifos: renderiza cada carácter una vez y ordena por tinta real."""

from dataclasses import dataclass

from functools import lru_cache

import numpy as np
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter

from .modes import BRAILLE_BASE, BRAILLE_BITS, EDGE_CHARS

CANDIDATES = " .'`^\",:;Il!i><~=+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Paletas de símbolos limpios (se ordenan por tinta medida). None = todos los candidatos.
BLOCK_CHARS = " ░▒▓█"
FULL_CHARS = CANDIDATES + "\\" + BLOCK_CHARS[1:]  # + "\" (bordes) y bloques (el espacio ya está)

BASE_PX = 16  # tamaño de fuente de pantalla; la exportación lo multiplica

PRESETS: dict[str, str | None] = {
    "Simple": " .:-=+*#%@",
    "Detallada": " .,:;-~=+!*ox#%&@",
    "Extendida": CANDIDATES,
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


def _to_array(img: QImage, cw: int, ch: int) -> np.ndarray:
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(ch, img.bytesPerLine())
    return buf[:, :cw].copy()


def render_glyphs(chars: str, family: str = "Menlo", px: int = BASE_PX) -> Atlas:
    """Renderiza los glifos EN EL ORDEN DADO (requiere QGuiApplication)."""
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
        maps.append(_to_array(img, cw, ch))
    return Atlas(chars, np.stack(maps), cw, ch)


def build_atlas(chars: str = FULL_CHARS, family: str = "Menlo", px: int = BASE_PX) -> Atlas:
    """Atlas con los glifos ordenados por tinta real (de menos a más)."""
    atlas = render_glyphs(chars, family, px)
    ink = atlas.bitmaps.reshape(len(chars), -1).sum(axis=1)
    order = np.argsort(ink, kind="stable")
    return Atlas("".join(chars[i] for i in order), atlas.bitmaps[order], atlas.cell_w, atlas.cell_h)


@lru_cache(maxsize=6)
def cell_size(px: int, family: str = "Menlo") -> tuple[int, int]:
    """(ancho, alto) de celda de la fuente a ese tamaño."""
    fm = QFontMetrics(_font(family, px))
    return fm.horizontalAdvance("M"), fm.height()


def rescale_atlas(atlas: Atlas, mode: str, px: int, family: str = "Menlo") -> Atlas:
    """El mismo atlas (mismos caracteres, mismo orden de índices) a otro tamaño de fuente.

    Imprescindible para exportar a mayor escala sin que los índices apunten a otro glifo.
    """
    cw, ch = cell_size(px, family)
    if mode == "Braille":
        return braille_atlas(cw, ch)
    if mode == "Bloques":
        return block_atlas(cw, ch)
    return render_glyphs(atlas.chars, family, px)


def pick_palette(atlas: Atlas, chars: str | None) -> Atlas:
    """Sub-atlas con los caracteres pedidos, ordenados por tinta (None = todos)."""
    if chars is None:
        return atlas
    keep = [i for i, c in enumerate(atlas.chars) if c in chars]
    return Atlas("".join(atlas.chars[i] for i in keep), atlas.bitmaps[keep],
                 atlas.cell_w, atlas.cell_h)


@lru_cache(maxsize=8)
def braille_atlas(cell_w: int, cell_h: int) -> Atlas:
    """256 glifos Braille dibujados de forma procedural (no dependen de la fuente)."""
    dx, dy = cell_w / 2, cell_h / 4
    r = 0.38 * min(dx, dy)
    maps = []
    for code in range(256):
        img = QImage(cell_w, cell_h, QImage.Format.Format_Grayscale8)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(0, 0, 0, 0))
        p.setBrush(QColor(255, 255, 255))
        for row in range(4):
            for col in range(2):
                if code & int(BRAILLE_BITS[row, col]):
                    p.drawEllipse((col + 0.5) * dx - r, (row + 0.5) * dy - r, 2 * r, 2 * r)
        p.end()
        maps.append(_to_array(img, cell_w, cell_h))
    chars = "".join(chr(BRAILLE_BASE + i) for i in range(256))
    return Atlas(chars, np.stack(maps), cell_w, cell_h)


@lru_cache(maxsize=8)
def block_atlas(cell_w: int, cell_h: int) -> Atlas:
    """Bloques " ░▒▓█" como relleno uniforme de toda la celda.

    Los glifos de la fuente no cubren la altura completa de la línea y dejan rayas
    horizontales; el relleno procedural las evita. Los caracteres solo se usan al exportar texto.
    """
    levels = np.linspace(0, 255, len(BLOCK_CHARS)).astype(np.uint8)
    maps = np.stack([np.full((cell_h, cell_w), v, np.uint8) for v in levels])
    return Atlas(BLOCK_CHARS, maps, cell_w, cell_h)


def edge_atlas(base: Atlas, shade_chars: str) -> Atlas:
    """Sombras ordenadas por tinta seguidas de los 4 glifos de borde (| / - \\)."""
    shade = pick_palette(base, shade_chars)
    pos = [base.chars.index(c) for c in EDGE_CHARS]
    return Atlas(shade.chars + EDGE_CHARS, np.concatenate([shade.bitmaps, base.bitmaps[pos]]),
                 base.cell_w, base.cell_h)


def atlas_for(mode: str, palette_name: str, base: Atlas) -> tuple[Atlas, int]:
    """Atlas del modo y nº de niveles de sombra (0 si el modo no usa niveles)."""
    if mode == "Braille":
        return braille_atlas(base.cell_w, base.cell_h), 0
    if mode == "Bloques":
        a = block_atlas(base.cell_w, base.cell_h)
        return a, len(a.chars)
    chars = PRESETS[palette_name]
    if mode == "Bordes":
        a = edge_atlas(base, chars)
        return a, len(a.chars) - len(EDGE_CHARS)
    a = pick_palette(base, chars)
    return a, len(a.chars)
