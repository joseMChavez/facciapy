"""Panel lateral de ajustes."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ..config import MODES, Settings
from ..glyphs import PRESETS
from ..render import THEMES

CAMERAS = 4  # índices 0..3 (sin sondear: abrir cámaras ajenas es intrusivo)


class ControlsPanel(QWidget):
    changed = Signal(str)  # nombre del ajuste modificado
    retry_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.setFixedWidth(270)
        self._labels: dict[str, QLabel] = {}
        root = QVBoxLayout(self)

        src = QFormLayout()
        self.camera = QComboBox()
        for i in range(CAMERAS):
            self.camera.addItem(f"Cámara {i}", i)
        self.camera.setToolTip("Índice de cámara. La del iPhone (Continuidad) suele ser otra.")
        self.camera.currentIndexChanged.connect(lambda _: self.changed.emit("camera_index"))
        self.mirror = QCheckBox("Espejo")
        self.mirror.toggled.connect(lambda _: self.changed.emit("mirror"))
        src.addRow("Cámara", self.camera)
        src.addRow("", self.mirror)
        root.addWidget(self._group("Fuente", src))

        img = QFormLayout()
        self.gamma = self._slider(img, "gamma", "Gamma", 40, 150, lambda v: f"{v / 100:.2f}",
                                  "Menor = imagen más clara")
        self.stability = self._slider(img, "smoothing", "Estabilidad", 0, 90, lambda v: f"{v}%",
                                      "Más estabilidad = menos parpadeo, algo de estela")
        root.addWidget(self._group("Imagen", img))

        sty = QFormLayout()
        self.palette = QComboBox()
        self.palette.addItems(PRESETS)
        self.palette.currentIndexChanged.connect(lambda _: self.changed.emit("palette_name"))
        self.theme = QComboBox()
        self.theme.addItems(THEMES)
        self.theme.currentIndexChanged.connect(lambda _: self.changed.emit("theme"))
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.setToolTip("ASCII · Bloques · Braille (8× más resolución) · Bordes")
        self.mode.currentIndexChanged.connect(lambda _: self._mode_changed())
        sty.addRow("Modo", self.mode)
        sty.addRow("Paleta", self.palette)
        sty.addRow("Color", self.theme)
        self.detail = self._slider(sty, "detail", "Detalle", 40, 240, lambda v: f"{v} col",
                                   "Columnas de caracteres")
        self.edges = self._slider(sty, "edge_sensitivity", "Bordes", 0, 100,
                                  lambda v: f"{v}%", "Sensibilidad de bordes (modo Bordes)")
        root.addWidget(self._group("Estilo", sty))

        self.retry = QPushButton("Reintentar cámara")
        self.retry.clicked.connect(self.retry_clicked)
        reset = QPushButton("Restablecer ajustes")
        reset.clicked.connect(self.reset_clicked)
        root.addWidget(self.retry)
        root.addWidget(reset)
        root.addStretch(1)
        self.set_values(settings)

    def _mode_changed(self) -> None:
        self._sync_enabled()
        self.changed.emit("mode")

    def _sync_enabled(self) -> None:
        mode = self.mode.currentText()
        self.palette.setEnabled(mode in ("ASCII", "Bordes"))
        self.edges.setEnabled(mode == "Bordes")

    def _group(self, title: str, layout: QFormLayout) -> QGroupBox:
        box = QGroupBox(title)
        box.setLayout(layout)
        return box

    def _slider(self, form: QFormLayout, name: str, title: str, lo: int, hi: int, fmt,
                tip: str) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setToolTip(tip)
        lab = QLabel()
        lab.setMinimumWidth(52)
        lab.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._labels[name] = lab

        def on(v: int) -> None:
            lab.setText(fmt(v))
            self.changed.emit(name)

        s.valueChanged.connect(on)
        s._fmt = fmt  # type: ignore[attr-defined]
        row = QHBoxLayout()
        row.addWidget(s)
        row.addWidget(lab)
        form.addRow(title, row)
        return s

    def set_values(self, st: Settings) -> None:
        widgets = [self.camera, self.mirror, self.gamma, self.stability, self.palette,
                   self.theme, self.detail, self.mode, self.edges]
        for w in widgets:
            w.blockSignals(True)
        self.camera.setCurrentIndex(max(0, min(CAMERAS - 1, st.camera_index)))
        self.mirror.setChecked(st.mirror)
        self.gamma.setValue(round(st.gamma * 100))
        self.stability.setValue(round((1 - st.smoothing) * 100))
        self.palette.setCurrentText(st.palette_name)
        self.theme.setCurrentText(st.theme)
        self.detail.setValue(st.detail)
        self.mode.setCurrentText(st.mode)
        self.edges.setValue(st.edge_sensitivity)
        for name, s in (("gamma", self.gamma), ("smoothing", self.stability),
                        ("detail", self.detail), ("edge_sensitivity", self.edges)):
            self._labels[name].setText(s._fmt(s.value()))  # type: ignore[attr-defined]
        for w in widgets:
            w.blockSignals(False)
        self._sync_enabled()

    def read_into(self, st: Settings) -> None:
        st.camera_index = int(self.camera.currentData())
        st.mirror = self.mirror.isChecked()
        st.gamma = self.gamma.value() / 100
        st.smoothing = 1 - self.stability.value() / 100
        st.palette_name = self.palette.currentText()
        st.theme = self.theme.currentText()
        st.detail = self.detail.value()
        st.mode = self.mode.currentText()
        st.edge_sensitivity = self.edges.value()
