"""Visor: imagen ASCII ajustada y centrada (opcionalmente junto al original), o un mensaje."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget


GAP = 8


def fit_rect(area: QRectF, iw: int, ih: int) -> QRectF:
    """Mayor rectángulo con la proporción iw:ih que cabe en `area`, centrado."""
    scale = min(area.width() / iw, area.height() / ih)
    w, h = iw * scale, ih * scale
    return QRectF(area.x() + (area.width() - w) / 2, area.y() + (area.height() - h) / 2, w, h)


def split_areas(area: QRectF) -> tuple[QRectF, QRectF]:
    """Mitades izquierda (original) y derecha (ASCII) con una separación."""
    half = (area.width() - GAP) / 2
    return (QRectF(area.x(), area.y(), half, area.height()),
            QRectF(area.x() + half + GAP, area.y(), half, area.height()))


class Viewport(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._original: QImage | None = None  # si no es None: vista dividida
        self._message = "Conectando con la cámara…"
        self._bg = QColor(13, 15, 18)
        self.setMinimumSize(320, 240)

    def set_background(self, rgb: tuple[int, int, int]) -> None:
        self._bg = QColor(*rgb)
        self.update()

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_original(self, image: QImage | None) -> None:
        self._original = image
        self.update()

    def set_message(self, text: str | None) -> None:
        self._message = text or ""
        self.update()

    @staticmethod
    def _label(p: QPainter, area: QRectF, text: str) -> None:
        p.setPen(QColor(150, 160, 175))
        p.drawText(area.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignTop
                   | Qt.AlignmentFlag.AlignLeft, text)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)
        if self._image is not None and not self._message:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            area = QRectF(self.rect())
            if self._original is not None:
                left, area = split_areas(area)
                p.drawImage(fit_rect(left, self._original.width(), self._original.height()),
                            self._original)
                self._label(p, left, "Original")
                self._label(p, area, "ASCII")
            p.drawImage(fit_rect(area, self._image.width(), self._image.height()), self._image)
        else:
            p.setPen(QColor(150, 160, 175))
            p.drawText(self.rect().adjusted(24, 24, -24, -24),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message)
        p.end()
