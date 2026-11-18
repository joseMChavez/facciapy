"""Diálogo de opciones de exportación."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QLabel, QSlider, QVBoxLayout)

from ..export import SCALES, ExportOptions, export_size, max_scale

PNG_LABEL = "PNG — sin pérdida (recomendado)"
JPEG_LABEL = "JPEG — con pérdida (solo por compatibilidad)"
HINT = ("PNG es la mejor opción para arte ASCII: caracteres nítidos y archivos mucho más "
        "pequeños (medido: 10–100× menos que JPEG, porque hay pocos colores y mucha repetición). "
        "JPEG deja halos alrededor de los caracteres; úsalo solo si el destino no acepta PNG.")


class ExportDialog(QDialog):
    def __init__(self, cols: int, rows: int, parent=None, defaults: ExportOptions | None = None):
        super().__init__(parent)
        self.setWindowTitle("Exportar imagen")
        self._cols, self._rows = cols, rows
        d = defaults or ExportOptions()

        self.fmt = QComboBox()
        self.fmt.addItem(PNG_LABEL, "PNG")
        self.fmt.addItem(JPEG_LABEL, "JPEG")
        self.fmt.setCurrentIndex(0 if d.fmt == "PNG" else 1)

        self.quality = QSlider(Qt.Orientation.Horizontal)
        self.quality.setRange(50, 100)
        self.quality.setValue(d.quality)
        self.quality_label = QLabel()

        top = max_scale(cols, rows)
        self.scale = QComboBox()
        for s in SCALES:
            self.scale.addItem(f"{s}×", s)
            if s > top:  # supera el límite de memoria
                self.scale.model().item(self.scale.count() - 1).setEnabled(False)
        self.scale.setCurrentIndex(max(0, min(SCALES.index(min(d.scale, top)), len(SCALES) - 1)))
        self.size_label = QLabel()
        self.text = QCheckBox("Guardar también el texto (.txt)")
        self.text.setChecked(d.save_text)

        hint = QLabel(HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8b95a7;")

        form = QFormLayout()
        form.addRow("Formato", self.fmt)
        form.addRow("Calidad JPEG", self.quality)
        form.addRow("", self.quality_label)
        form.addRow("Escala", self.scale)
        form.addRow("", self.size_label)
        form.addRow("", self.text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(hint)
        root.addWidget(buttons)
        self.setMinimumWidth(420)

        for sig in (self.fmt.currentIndexChanged, self.scale.currentIndexChanged,
                    self.quality.valueChanged):
            sig.connect(lambda *_: self._refresh())
        self._refresh()

    def _refresh(self) -> None:
        jpeg = self.fmt.currentData() == "JPEG"
        self.quality.setEnabled(jpeg)
        self.quality_label.setText(f"{self.quality.value()}" if jpeg else "(no aplica a PNG)")
        w, h = export_size(self._cols, self._rows, int(self.scale.currentData()))
        self.size_label.setText(f"Resultado: {w}×{h} px ({w * h / 1e6:.1f} MP)")

    def options(self) -> ExportOptions:
        return ExportOptions(fmt=self.fmt.currentData(), quality=self.quality.value(),
                             scale=int(self.scale.currentData()), save_text=self.text.isChecked())
