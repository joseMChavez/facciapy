"""Imágenes estáticas como fuente alternativa a la cámara."""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QImageReader

from .capture import State

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")
MAX_SIDE = 2000  # más resolución no aporta: la rejilla ASCII es de unos cientos de celdas


def load_image(path: str | Path) -> np.ndarray:
    """Carga una imagen como BGR uint8. Lanza ValueError si no se puede leer.

    Usa el lector de Qt: aplica la orientación EXIF (fotos de móvil), soporta rutas
    con acentos y fuentes de 8/16 bits o con transparencia (se compone sobre negro).
    """
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    img = reader.read()
    if img.isNull():
        raise ValueError(f"No se pudo abrir la imagen: {reader.errorString()}")

    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = img.height(), img.width()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())
    rgba = buf[:, : w * 4].reshape(h, w, 4)
    alpha = rgba[..., 3:4].astype(np.uint16)
    rgb = (rgba[..., :3].astype(np.uint16) * alpha // 255).astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])

    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return bgr


class StaticImageSource:
    """Misma interfaz que CameraSource: start / latest / stop / state."""

    is_static = True

    def __init__(self, frame: np.ndarray, name: str = "imagen"):
        self.frame = frame
        self.name = name
        self.state = State.LIVE

    def start(self) -> None: ...

    def latest(self) -> tuple[int, np.ndarray]:
        return 1, self.frame

    def stop(self, timeout: float = 0) -> None: ...
