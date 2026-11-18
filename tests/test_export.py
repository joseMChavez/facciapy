import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from facia_app.capture import State  # noqa: E402
from facia_app.config import MODES, Settings  # noqa: E402
from facia_app.export import (ExportOptions, ensure_suffix, export_size, max_scale,  # noqa: E402
                              render_export, save_image)
from facia_app.glyphs import atlas_for, build_atlas  # noqa: E402
from facia_app.ui.export_dialog import ExportDialog  # noqa: E402
from facia_app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def base(app):
    return build_atlas()


def grid(levels, rows=12, cols=30):
    rng = np.random.default_rng(0)
    return rng.integers(0, levels, (rows, cols), dtype=np.uint8)


def test_scale1_matches_screen_render(base):
    """A escala 1 la exportación debe coincidir píxel a píxel con lo que se ve en pantalla."""
    from facia_app.render import compose, to_qimage
    atlas, levels = atlas_for("ASCII", "Simple", base)
    idx = grid(levels)
    out = render_export(idx, None, atlas, "ASCII", "Mono", 1)
    ref = to_qimage(compose(idx, atlas), "Mono")
    assert out == ref


@pytest.mark.parametrize("mode", MODES)
def test_scale_changes_size_and_keeps_glyph_mapping(base, mode):
    atlas, levels = atlas_for(mode, "Simple", base)
    idx = grid(len(atlas.chars) if mode != "Bordes" else levels + 4)
    a = render_export(idx, None, atlas, mode, "Mono", 1)
    b = render_export(idx, None, atlas, mode, "Mono", 3)
    assert (b.width(), b.height()) == export_size(idx.shape[1], idx.shape[0], 3)
    assert b.width() > 2 * a.width()
    # mismo contenido: el brillo de cada celda debe correlacionar entre escalas.
    # Si los índices apuntaran a otro glifo (orden distinto del atlas) la correlación caería.
    rows, cols = idx.shape

    def cell_means(im: QImage) -> np.ndarray:
        g = im.convertToFormat(QImage.Format.Format_Grayscale8)
        arr = np.frombuffer(g.constBits(), np.uint8).reshape(g.height(), g.bytesPerLine())
        arr = arr[:, : g.width()].astype(float)
        ch, cw = g.height() // rows, g.width() // cols
        return arr[: rows * ch, : cols * cw].reshape(rows, ch, cols, cw).mean(axis=(1, 3))

    corr = np.corrcoef(cell_means(a).ravel(), cell_means(b).ravel())[0, 1]
    assert corr > 0.97, f"correlación {corr:.3f}"


def test_color_export_rgb(base):
    atlas, levels = atlas_for("ASCII", "Simple", base)
    idx = grid(levels)
    colors = np.full((*idx.shape, 3), (255, 0, 0), np.uint8)
    img = render_export(idx, colors, atlas, "ASCII", "Color real", 2)
    assert img.format() == QImage.Format.Format_RGB888
    px = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    px = px[:, : img.width() * 3].reshape(img.height(), img.width(), 3)
    assert px[..., 0].max() > 0 and px[..., 1].max() == 0 and px[..., 2].max() == 0  # solo rojo


def test_png_is_lossless_and_jpeg_quality_matters(base, tmp_path):
    atlas, levels = atlas_for("ASCII", "Simple", base)
    img = render_export(grid(levels, 30, 80), None, atlas, "ASCII", "Mono", 2)
    png, lo, hi = tmp_path / "a.png", tmp_path / "lo.jpg", tmp_path / "hi.jpg"
    save_image(img, str(png), "PNG")
    save_image(img, str(lo), "JPEG", 30)
    save_image(img, str(hi), "JPEG", 95)
    assert QImage(str(png)).convertToFormat(QImage.Format.Format_RGB32) == \
        img.convertToFormat(QImage.Format.Format_RGB32)  # sin pérdida
    assert hi.read_bytes()[:2] == b"\xff\xd8" and lo.stat().st_size < hi.stat().st_size


def test_save_errors(base, tmp_path):
    atlas, levels = atlas_for("ASCII", "Simple", base)
    img = render_export(grid(levels), None, atlas, "ASCII", "Mono", 1)
    with pytest.raises(ValueError):
        save_image(img, str(tmp_path / "x.gif"), "GIF")
    with pytest.raises(OSError):
        save_image(img, str(tmp_path / "no_existe" / "x.png"), "PNG")


def test_ensure_suffix():
    assert ensure_suffix("/a/b", "PNG") == "/a/b.png"
    assert ensure_suffix("/a/b.JPEG", "JPEG") == "/a/b.JPEG"
    assert ensure_suffix("/a/b.png", "JPEG") == "/a/b.jpg"


def test_max_scale_limits_memory(app):
    assert max_scale(120, 35) == 4
    assert max_scale(240, 70) >= 2
    assert max_scale(3000, 1000) == 1


def test_dialog_options_and_limits(app):
    dlg = ExportDialog(120, 35)
    assert dlg.options() == ExportOptions(fmt="PNG", quality=92, scale=2, save_text=False)
    assert not dlg.quality.isEnabled()
    dlg.fmt.setCurrentIndex(1)
    assert dlg.quality.isEnabled() and dlg.options().fmt == "JPEG"
    big = ExportDialog(3000, 1000)
    assert not big.scale.model().item(3).isEnabled() and big.options().scale == 1


class FakeCam:
    state = State.LIVE

    def __init__(self, _src=0): ...
    def start(self): ...
    def stop(self, timeout=0): ...

    def latest(self):
        g = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (360, 1))
        return 1, np.dstack([g, g, g])


@pytest.mark.parametrize("theme", ["Mono", "Color real"])
def test_window_exports_current_frame(app, tmp_path, theme):
    win = MainWindow(Settings(detail=60, theme=theme), camera_factory=FakeCam)
    win._tick()
    assert (win._colors is not None) == (theme == "Color real")
    rows, cols = win._idx.shape
    img = render_export(win._idx, win._colors, win._atlas, win.settings.mode, theme, 2)
    out = tmp_path / "e.png"
    save_image(img, str(out), "PNG")
    assert QImage(str(out)).size().width() == export_size(cols, rows, 2)[0]
    win.close()
