"""Visor: dibuja la imagen ASCII ajustada y centrada, o un mensaje."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget


class Viewport(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._message = "Conectando con la cámara…"
        self._bg = QColor(13, 15, 18)
        self.setMinimumSize(320, 240)

    def set_background(self, rgb: tuple[int, int, int]) -> None:
        self._bg = QColor(*rgb)
        self.update()

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_message(self, text: str | None) -> None:
        self._message = text or ""
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)
        if self._image is not None and not self._message:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            iw, ih = self._image.width(), self._image.height()
            scale = min(self.width() / iw, self.height() / ih)
            w, h = iw * scale, ih * scale
            p.drawImage(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h),
                        self._image)
        else:
            p.setPen(QColor(150, 160, 175))
            p.drawText(self.rect().adjusted(24, 24, -24, -24),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message)
        p.end()
