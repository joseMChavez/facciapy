import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from facia_app.capture import State  # noqa: E402
from facia_app.config import Settings  # noqa: E402
from facia_app.glyphs import PRESETS, build_atlas, pick_palette  # noqa: E402
from facia_app.render import compose, to_qimage, to_text  # noqa: E402
from facia_app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atlas(app):
    return build_atlas()


def test_atlas_sorted_by_ink(atlas):
    ink = atlas.bitmaps.reshape(len(atlas.chars), -1).sum(axis=1)
    assert (np.diff(ink) >= 0).all()
    assert atlas.chars[0] == " "


@pytest.mark.parametrize("name", ["Simple", "Detallada"])
def test_palette_strictly_increasing_ink(atlas, name):
    pal = pick_palette(atlas, PRESETS[name])
    ink = pal.bitmaps.reshape(len(pal.chars), -1).sum(axis=1)
    assert len(pal.chars) == len(set(PRESETS[name])) and (np.diff(ink) > 0).all()


def test_compose_and_text(atlas):
    pal = pick_palette(atlas, PRESETS["Simple"])
    idx = np.tile(np.arange(10, dtype=np.uint8), (3, 1))
    gray = compose(idx, pal)
    assert gray.shape == (3 * pal.cell_h, 10 * pal.cell_w)
    img = to_qimage(gray, "Mono")
    assert (img.width(), img.height()) == (gray.shape[1], gray.shape[0])
    assert to_text(idx, pal).count("\n") == 2


class FakeCam:
    state = State.LIVE

    def __init__(self, _src=0):
        self.stopped = False

    def start(self): ...
    def stop(self, timeout=0): self.stopped = True

    def latest(self):
        g = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (360, 1))
        return 1, np.dstack([g, g, g])


def test_window_renders_frame(app):
    win = MainWindow(Settings(detail=80), camera_factory=FakeCam)
    win._tick()
    assert win._idx is not None and win._idx.shape[1] == 80
    win.panel.palette.setCurrentText("Detallada")
    assert len(win._atlas.chars) == len(PRESETS["Detallada"])
    win.close()
