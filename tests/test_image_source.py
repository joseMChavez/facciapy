import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from facia_app.capture import State  # noqa: E402
from facia_app.config import Settings  # noqa: E402
from facia_app.image_source import MAX_SIDE, StaticImageSource, load_image  # noqa: E402
from facia_app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def asym(h=120, w=200):
    """Imagen asimétrica: mitad izquierda oscura, derecha clara."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, w // 2 :] = 220
    return img


def test_load_png_bgr_order(app, tmp_path):
    img = np.zeros((20, 30, 3), np.uint8)
    img[..., 2] = 255  # rojo en BGR
    cv2.imwrite(str(tmp_path / "r.png"), img)
    out = load_image(tmp_path / "r.png")
    assert out.shape == (20, 30, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (0, 0, 255)


def test_load_jpeg_and_gray(app, tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), asym())
    cv2.imwrite(str(tmp_path / "g.png"), asym()[..., 0])
    assert load_image(tmp_path / "a.jpg").shape == (120, 200, 3)
    assert load_image(tmp_path / "g.png").shape == (120, 200, 3)


def test_load_16bit(app, tmp_path):
    img = np.full((10, 10, 3), 65535, np.uint16)
    cv2.imwrite(str(tmp_path / "h.png"), img)
    assert load_image(tmp_path / "h.png").min() >= 250


def test_alpha_composited_over_black(app, tmp_path):
    img = np.zeros((10, 10, 4), np.uint8)
    img[..., :3] = 255
    img[:, 5:, 3] = 255  # mitad transparente
    cv2.imwrite(str(tmp_path / "t.png"), img)
    out = load_image(tmp_path / "t.png")
    assert out[0, 0].max() == 0 and out[0, 9].min() == 255


def test_unicode_path(app, tmp_path):
    p = tmp_path / "niño_ñandú.png"
    cv2.imwrite(str(p), asym())
    assert load_image(p).shape == (120, 200, 3)


def test_large_image_downscaled(app, tmp_path):
    cv2.imwrite(str(tmp_path / "big.png"), np.zeros((500, 4000, 3), np.uint8))
    out = load_image(tmp_path / "big.png")
    assert max(out.shape[:2]) == MAX_SIDE and out.shape[1] / out.shape[0] == 8


def test_missing_and_corrupt(app, tmp_path):
    with pytest.raises(ValueError):
        load_image(tmp_path / "no_existe.png")
    bad = tmp_path / "mala.png"
    bad.write_bytes(b"esto no es una imagen")
    with pytest.raises(ValueError):
        load_image(bad)


def test_static_source_interface():
    src = StaticImageSource(asym(), "x.png")
    src.start()
    seq, frame = src.latest()
    assert seq == 1 and frame.shape == (120, 200, 3) and src.state is State.LIVE
    src.stop()


class FakeCam:
    state = State.LIVE

    def __init__(self, _src=0):
        self.stopped = False

    def start(self): ...
    def stop(self, timeout=0): self.stopped = True
    def latest(self): return 1, asym(72, 128)


def test_window_loads_image_and_rerenders(app, tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), asym())
    win = MainWindow(Settings(detail=60, mirror=True), camera_factory=FakeCam)
    cam = win.cam
    assert win.load_image_file(str(tmp_path / "a.png"))
    assert cam.stopped and win._is_static
    win._tick()
    first = win._idx.copy()
    assert first[:, 0].max() < first[:, -1].min()  # izquierda oscura, derecha clara
    # espejo activado no debe voltear un archivo
    win.panel.mirror.setChecked(False); win.panel.mirror.setChecked(True)
    win._tick()
    assert (win._idx == first).all()
    # un cambio de ajustes re-renderiza aunque la fuente sea estática (seq no cambia)
    win.panel.detail.setValue(90)
    win._tick()
    assert win._idx.shape[1] == 90
    win.close()


def test_failed_load_keeps_source(app, tmp_path):
    win = MainWindow(Settings(), camera_factory=FakeCam)
    cam = win.cam
    assert not win.load_image_file(str(tmp_path / "nada.png"))
    assert win.cam is cam and not cam.stopped
    win.close()


def test_use_camera_returns_from_image(app, tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), asym())
    win = MainWindow(Settings(), camera_factory=FakeCam)
    win.load_image_file(str(tmp_path / "a.png"))
    win.use_camera()
    assert isinstance(win.cam, FakeCam) and not win._is_static
    win.close()


def test_dropped_image_filter(app):
    class U:
        def __init__(s, p): s.p = p
        def isLocalFile(s): return True
        def toLocalFile(s): return s.p

    class Ev:
        def __init__(s, paths): s.paths = paths
        def mimeData(s):
            m = type("M", (), {})()
            m.urls = lambda: [U(p) for p in s.paths]
            return m

    assert MainWindow._dropped_image(Ev(["/a/doc.txt", "/a/foto.JPG"])) == "/a/foto.JPG"
    assert MainWindow._dropped_image(Ev(["/a/doc.txt"])) is None


def _jpeg_with_orientation(img: np.ndarray, orientation: int) -> bytes:
    """Inserta un segmento EXIF mínimo (solo la etiqueta Orientation)."""
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    tiff = (b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00\x01"
            + b"\x01\x12\x00\x03\x00\x00\x00\x01" + orientation.to_bytes(2, "big") + b"\x00\x00"
            + b"\x00\x00\x00\x00")
    body = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body
    data = enc.tobytes()
    return data[:2] + app1 + data[2:]


def test_exif_orientation_applied(app, tmp_path):
    p = tmp_path / "movil.jpg"
    p.write_bytes(_jpeg_with_orientation(asym(100, 200), 6))  # 6 = girar 90° horario
    out = load_image(p)
    assert out.shape[:2] == (200, 100)  # alto y ancho intercambiados
