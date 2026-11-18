"""Exportar el arte a imagen (PNG o JPEG) a la escala que se quiera."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from .glyphs import BASE_PX, Atlas, cell_size, rescale_atlas
from .render import COLOR_THEME, compose, to_qimage, to_qimage_color

MAX_PIXELS = 40_000_000  # ~160 MB en RGB; más allá, la memoria se dispara
SCALES = (1, 2, 3, 4)
FORMATS = {"PNG": ".png", "JPEG": ".jpg"}


@dataclass
class ExportOptions:
    fmt: str = "PNG"
    quality: int = 92  # solo JPEG
    scale: int = 2
    save_text: bool = False


def export_size(cols: int, rows: int, scale: int) -> tuple[int, int]:
    cw, ch = cell_size(BASE_PX * scale)
    return cols * cw, rows * ch


def max_scale(cols: int, rows: int) -> int:
    """Mayor escala de SCALES que no supera MAX_PIXELS (mínimo 1)."""
    ok = [s for s in SCALES if np.prod(export_size(cols, rows, s)) <= MAX_PIXELS]
    return max(ok) if ok else 1


def render_export(idx: np.ndarray, colors: np.ndarray | None, atlas: Atlas, mode: str,
                  theme: str, scale: int) -> QImage:
    """Re-renderiza la rejilla con glifos de mayor tamaño (texto nítido, no un reescalado)."""
    hi = rescale_atlas(atlas, mode, BASE_PX * scale)
    gray = compose(idx, hi)
    if theme == COLOR_THEME and colors is not None:
        return to_qimage_color(gray, colors, hi.cell_w, hi.cell_h)
    return to_qimage(gray, theme)


def ensure_suffix(path: str, fmt: str) -> str:
    """Añade la extensión del formato si falta o no corresponde (.jpg/.jpeg valen para JPEG)."""
    p = Path(path)
    valid = (".jpg", ".jpeg") if fmt == "JPEG" else (".png",)
    return str(p) if p.suffix.lower() in valid else str(p.with_suffix(FORMATS[fmt]))


def save_image(img: QImage, path: str, fmt: str = "PNG", quality: int = 92) -> None:
    """Guarda la imagen. PNG es sin pérdida (calidad 0 = máxima compresión); JPEG con pérdida."""
    if fmt not in FORMATS:
        raise ValueError(f"Formato no soportado: {fmt}")
    ok = img.save(path, fmt, 0 if fmt == "PNG" else int(np.clip(quality, 1, 100)))
    if not ok:
        raise OSError(f"No se pudo guardar en {path}")
