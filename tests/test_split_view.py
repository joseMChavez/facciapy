import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from facia_app.capture import State  # noqa: E402
from facia_app.config import Settings  # noqa: E402
from facia_app.ui.main_window import MainWindow, load_settings, save_settings  # noqa: E402
from facia_app.ui.viewport import GAP, fit_rect, split_areas  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_fit_rect_keeps_aspect_and_centers():
    r = fit_rect(QRectF(0, 0, 400, 400), 200, 100)
    assert (r.width(), r.height()) == (400, 200) and r.y() == 100
    r = fit_rect(QRectF(10, 20, 300, 600), 100, 100)
    assert r.width() == r.height() == 300 and r.x() == 10


def test_split_areas_do_not_overlap():
    left, right = split_areas(QRectF(0, 0, 1000, 500))
    assert left.width() == right.width() == (1000 - GAP) / 2
    assert right.x() - (left.x() + left.width()) == GAP and right.right() == 1000


class FakeCam:
    state = State.LIVE

    def __init__(self, _src=0): ...
    def start(self): ...
    def stop(self, timeout=0): ...

    def latest(self):
        f = np.zeros((720, 1280, 3), np.uint8)
        f[:, 640:] = (30, 120, 220)  # mitad derecha anaranjada (BGR)
        return 1, f


def test_split_toggle_shows_and_hides_original(app):
    win = MainWindow(Settings(detail=60, mirror=False), camera_factory=FakeCam)
    win._tick()
    assert win.viewport._original is None
    win.panel.split.setChecked(True)
    win._tick()
    orig = win.viewport._original
    assert orig is not None and orig.width() == 640 and orig.height() == 360  # reducido
    win.panel.split.setChecked(False)
    win._tick()
    assert win.viewport._original is None
    win.close()


def test_original_matches_processed_frame(app):
    """Con espejo activo, el original mostrado debe ser el espejado (el mismo que se convierte)."""
    win = MainWindow(Settings(detail=60, mirror=True, split_view=True), camera_factory=FakeCam)
    win._tick()
    img = win.viewport._original
    left, right = img.pixelColor(10, 10), img.pixelColor(img.width() - 10, 10)
    assert left.red() > 150 and right.red() < 20  # naranja a la izquierda tras el espejo
    win.close()


def test_toggle_shortcut_and_persistence(app):
    win = MainWindow(Settings(), camera_factory=FakeCam)
    win._toggle_split()
    assert win.settings.split_view is True
    save_settings(win.settings)
    win.close()
    assert load_settings().split_view is True
    st = Settings()
    save_settings(st)  # restaurar para no contaminar otras ejecuciones
    assert load_settings().split_view is False
