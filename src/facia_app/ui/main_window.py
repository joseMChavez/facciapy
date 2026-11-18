"""Ventana principal."""

import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QStatusBar,
                               QWidget)

from ..capture import CameraSource, State
from ..config import Settings
from ..engine import Engine, fit_grid
from ..glyphs import PRESETS, Atlas, build_atlas, pick_palette
from ..render import THEMES, compose, to_qimage, to_text
from .controls import ControlsPanel
from .viewport import Viewport

PERSISTED = ("camera_index", "mirror", "gamma", "smoothing", "detail", "palette_name", "theme")
TICK_MS = 33

PERMISSION_HELP = (
    "No se pudo abrir la cámara.\n\n"
    "1. Ajustes del Sistema → Privacidad y seguridad → Cámara\n"
    "2. Activa el permiso para la app desde la que lanzaste esto (Terminal, VS Code…)\n"
    "3. Pulsa «Reintentar cámara»\n\n"
    "Si usas la cámara en otra app, prueba otro índice."
)

STATE_TEXT = {
    State.CONNECTING: "Conectando",
    State.LIVE: "En vivo",
    State.RECONNECTING: "Reconectando",
    State.FAILED: "Sin cámara",
}

STYLE = """
QMainWindow, QWidget { background: #14171c; color: #d8dee9; }
QGroupBox { border: 1px solid #2a2f38; border-radius: 6px; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8b95a7; }
QComboBox, QPushButton { background: #1d222a; border: 1px solid #2f3542; border-radius: 4px;
  padding: 4px 8px; }
QPushButton:hover { background: #262c36; }
QStatusBar { color: #8b95a7; }
"""


def load_settings() -> Settings:
    st, qs = Settings(), QSettings("facia-app", "facia-app")
    for f in fields(st):
        if f.name not in PERSISTED or not qs.contains(f.name):
            continue
        raw, cur = qs.value(f.name), getattr(st, f.name)
        try:
            val = str(raw).lower() == "true" if isinstance(cur, bool) else type(cur)(raw)
        except (TypeError, ValueError):
            continue
        setattr(st, f.name, val)
    if st.palette_name not in PRESETS:
        st.palette_name = Settings.palette_name
    if st.theme not in THEMES:
        st.theme = Settings.theme
    return st


def save_settings(st: Settings) -> None:
    qs = QSettings("facia-app", "facia-app")
    for name in PERSISTED:
        qs.setValue(name, getattr(st, name))


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, camera_factory=CameraSource) -> None:
        super().__init__()
        self.setWindowTitle("facia")
        self.resize(1200, 700)
        self.setStyleSheet(STYLE)
        self.settings = settings
        self._factory = camera_factory

        self._full_atlas = build_atlas()
        self._atlas: Atlas = self._full_atlas
        self._engine: Engine | None = None
        self._idx: np.ndarray | None = None
        self._image = None
        self._paused = False
        self._last_seq = 0
        self._frames = 0
        self._fps = 0.0
        self._fps_t = time.perf_counter()
        self._ms = 0.0
        self._res = ""

        self.viewport = Viewport()
        self.panel = ControlsPanel(settings)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.viewport, 1)
        lay.addWidget(self.panel)
        self.setCentralWidget(central)
        self.status_label = QLabel("")
        bar = QStatusBar()
        bar.addWidget(self.status_label, 1)
        self.setStatusBar(bar)

        self.panel.changed.connect(self._on_changed)
        self.panel.retry_clicked.connect(self._restart_camera)
        self.panel.reset_clicked.connect(self._reset)
        for key, fn in (("Space", self._toggle_pause), ("S", self.save_files),
                        ("F", self._toggle_fullscreen), ("P", self._toggle_panel),
                        (QKeySequence.StandardKey.Copy, self.copy_text)):
            QShortcut(QKeySequence(key), self, activated=fn)

        self._rebuild_pipeline()
        self.cam = self._factory(settings.camera_index)
        self.cam.start()
        self.timer = QTimer(self, interval=TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # --- configuración ---------------------------------------------------
    def _rebuild_pipeline(self) -> None:
        st = self.settings
        self._atlas = pick_palette(self._full_atlas, PRESETS[st.palette_name])
        self._engine = Engine(len(self._atlas.chars), st.gamma, st.smoothing)
        self.viewport.set_background(THEMES[st.theme][0])

    def _on_changed(self, name: str) -> None:
        self.panel.read_into(self.settings)
        if name == "camera_index":
            self._restart_camera()
        else:
            self._rebuild_pipeline()

    def _reset(self) -> None:
        self.settings = Settings()
        self.panel.set_values(self.settings)
        self._rebuild_pipeline()
        self._restart_camera()

    def _restart_camera(self) -> None:
        self.cam.stop()
        self._last_seq, self._res = 0, ""
        self.viewport.set_message("Conectando con la cámara…")
        self.cam = self._factory(self.settings.camera_index)
        self.cam.start()

    # --- bucle -------------------------------------------------------------
    def _tick(self) -> None:
        seq, frame = self.cam.latest()
        state = self.cam.state
        if state is State.FAILED and frame is None:
            self.viewport.set_message(PERMISSION_HELP)
        elif state is State.RECONNECTING:
            self.viewport.set_message("Reconectando con la cámara…")
        elif frame is None:
            self.viewport.set_message("Conectando con la cámara…")
        if frame is not None and seq != self._last_seq and not self._paused:
            self._last_seq = seq
            self._process(frame)
        self._update_status(state)

    def _process(self, frame: np.ndarray) -> None:
        t0 = time.perf_counter()
        st = self.settings
        if st.mirror:
            frame = frame[:, ::-1]
        h, w = frame.shape[:2]
        self._res = f"{w}×{h}"
        cols, rows = fit_grid((h, w), st.detail, 10**6, self._atlas.aspect)
        idx = self._engine.process(np.ascontiguousarray(frame), cols, rows)
        self._idx = idx
        self._image = to_qimage(compose(idx, self._atlas), st.theme)
        self.viewport.set_message(None)
        self.viewport.set_image(self._image)
        self._ms = 0.8 * self._ms + 0.2 * (time.perf_counter() - t0) * 1000
        self._frames += 1

    def _update_status(self, state: State) -> None:
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps = self._frames / (now - self._fps_t)
            self._frames, self._fps_t = 0, now
        dot = "●" if state is State.LIVE else "○"
        parts = [f"{dot} {STATE_TEXT[state]}" + (" (pausa)" if self._paused else "")]
        if self._res:
            parts.append(f"Cámara {self.settings.camera_index} {self._res}")
        if self._idx is not None:
            parts.append(f"{self._idx.shape[1]}×{self._idx.shape[0]} celdas")
            parts.append(f"{self._fps:.0f} fps · {self._ms:.1f} ms")
        self.status_label.setText("  ·  ".join(parts))

    # --- acciones ----------------------------------------------------------
    def _toggle_pause(self) -> None:
        self._paused = not self._paused

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _toggle_panel(self) -> None:
        self.panel.setVisible(not self.panel.isVisible())

    def copy_text(self) -> None:
        if self._idx is not None:
            QGuiApplication.clipboard().setText(to_text(self._idx, self._atlas))
            self.statusBar().showMessage("ASCII copiado", 2000)

    def save_files(self) -> None:
        if self._idx is None or self._image is None:
            return
        default = str(Path.home() / "Pictures" / f"facia_{datetime.now():%Y%m%d_%H%M%S}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Guardar", default, "PNG (*.png)")
        if not path:
            return
        self._image.save(path)
        Path(path).with_suffix(".txt").write_text(to_text(self._idx, self._atlas),
                                                  encoding="utf-8")
        self.statusBar().showMessage(f"Guardado: {path} (+ .txt)", 4000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        save_settings(self.settings)
        self.cam.stop()
        super().closeEvent(event)
