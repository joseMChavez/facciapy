import numpy as np
import pytest

from facia_app import face
from facia_app.face import DET_WIDTH, Face, FaceTracker, model_path

W, H = 1280, 720
K = W / DET_WIDTH  # factor de la miniatura de detección al frame


class Scripted:
    """Detector falso: devuelve la lista de rostros configurada (en coords de miniatura)."""

    def __init__(self, faces=()):
        self.faces, self.calls = list(faces), 0

    def detect(self, bgr):
        self.calls += 1
        assert bgr.shape[1] == DET_WIDTH
        return self.faces


def frame():
    return np.zeros((H, W, 3), np.uint8)


def face_at(cx, cy, size=40):  # en coordenadas de la miniatura
    return Face(cx - size / 2, cy - size / 2, size, size)


def test_no_face_returns_full_frame():
    t = FaceTracker(Scripted())
    out = t.crop(frame())
    assert out.shape == (H, W, 3) and not t.has_face


def test_first_detection_snaps_to_face_and_keeps_aspect():
    det = Scripted([face_at(100, 60)])
    t = FaceTracker(det)
    out = t.crop(frame())
    h, w = out.shape[:2]
    assert h < H and abs(w / h - W / H) < 0.02 and t.has_face
    # el crop contiene el centro del rostro (en coordenadas de frame)
    fx, fy = 100 * K, 60 * K
    # recuperar la posición del crop comparando con un frame marcado
    f = frame()
    f[int(fy) - 3 : int(fy) + 3, int(fx) - 3 : int(fx) + 3] = 255
    t2 = FaceTracker(Scripted([face_at(100, 60)]))
    assert t2.crop(f).max() == 255


@pytest.mark.parametrize("cx,cy", [(5, 5), (315, 175), (160, 5), (5, 175), (160, 90)])
def test_crop_always_inside_frame(cx, cy):
    t = FaceTracker(Scripted([face_at(cx, cy, 60)]))
    out = t.crop(frame())
    h, w = out.shape[:2]
    assert 0 < h <= H and 0 < w <= W
    assert abs(w / h - W / H) < 0.02


def test_detects_only_every_n_frames():
    det = Scripted([face_at(160, 90)])
    t = FaceTracker(det, every=3)
    for _ in range(9):
        t.crop(frame())
    assert det.calls == 3  # frames 0, 3 y 6


def test_follows_smoothly_after_first_lock():
    det = Scripted([face_at(80, 90)])
    t = FaceTracker(det, every=1)
    f = frame()
    t.crop(f)
    det.faces = [face_at(240, 90)]  # el rostro salta a la derecha
    xs = []
    for _ in range(12):
        out = t.crop(f)
        xs.append(t._state[0])
    assert xs[0] > 80 * K and all(b >= a for a, b in zip(xs, xs[1:]))
    assert xs[0] < xs[-1] < 240 * K + 1  # avanza gradualmente, sin saltar
    assert (xs[1] - xs[0]) < (240 - 80) * K * 0.3  # el primer paso es parcial


def test_picks_largest_face():
    det = Scripted([face_at(50, 90, 20), face_at(250, 90, 80)])
    t = FaceTracker(det)
    t.crop(frame())
    assert abs(t._target[0] - 250 * K) < 1


def test_returns_to_full_frame_when_face_is_lost():
    det = Scripted([face_at(160, 90, 50)])
    t = FaceTracker(det, every=1, lost_after=3, smoothing=0.5)
    f = frame()
    assert t.crop(f).shape[0] < H
    det.faces = []
    sizes = [t.crop(f).shape[0] for _ in range(40)]
    assert sizes[-1] == H and sizes[-1] >= sizes[0]
    assert not t.has_face


def test_does_not_zoom_in_too_much():
    t = FaceTracker(Scripted([face_at(160, 90, 4)]))  # rostro minúsculo
    assert t.crop(frame()).shape[0] >= H * face.MIN_CROP - 1


def test_reset_forgets_state():
    t = FaceTracker(Scripted([face_at(160, 90)]))
    t.crop(frame())
    t.reset()
    t.detector.faces = []
    assert t.crop(frame()).shape[:2] == (H, W)


def test_model_path_env_and_missing(tmp_path, monkeypatch):
    fake = tmp_path / "m.onnx"
    fake.write_bytes(b"x")
    monkeypatch.setenv(face.ENV_VAR, str(fake))
    assert model_path() == fake
    monkeypatch.setenv(face.ENV_VAR, str(tmp_path / "nada.onnx"))
    monkeypatch.setattr(face, "__file__", str(tmp_path / "a" / "b" / "c.py"))
    assert model_path() is None


@pytest.mark.skipif(face.model_path() is None, reason="modelo YuNet no descargado")
def test_real_yunet_loads_and_handles_blank():
    det = face.YuNetDetector(face.model_path())
    assert det.detect(np.zeros((180, 320, 3), np.uint8)) == []
    t = face.make_tracker()
    assert t is not None and t.crop(frame()).shape == (H, W, 3)


# ---------------- integración con la ventana ----------------
import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from facia_app.capture import State  # noqa: E402
from facia_app.config import Settings  # noqa: E402
from facia_app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Cam:
    state = State.LIVE

    def __init__(self, _src=0): ...
    def start(self): ...
    def stop(self, timeout=0): ...

    def latest(self):
        f = np.zeros((H, W, 3), np.uint8)
        f[200:260, 800:860] = 255  # "rostro" brillante pequeño
        return 1, f


def test_window_crops_to_face(app, monkeypatch):
    fx, fy = 830 / K, 230 / K
    monkeypatch.setattr(face, "make_tracker",
                        lambda: FaceTracker(Scripted([face_at(fx, fy, 22)])))
    win = MainWindow(Settings(detail=60, mirror=False, face_follow=True), camera_factory=Cam)
    win.panel.face.setEnabled(True)
    win._tick()
    assert win._tracker is not None and win._tracker.has_face
    assert win._res and int(win._res.split("×")[1]) < H  # el frame procesado es el recorte
    assert win._idx.max() > 0  # el rostro brillante sigue dentro del encuadre
    win.close()


def test_missing_model_disables_feature_with_message(app, monkeypatch):
    monkeypatch.setattr(face, "make_tracker", lambda: None)
    win = MainWindow(Settings(detail=60, face_follow=True), camera_factory=Cam)
    win._tick()
    assert win.settings.face_follow is False and not win.panel.face.isChecked()
    assert "modelo" in win.statusBar().currentMessage()
    assert win._idx is not None  # sigue renderizando el frame completo
    win.close()


def test_model_load_error_is_reported_not_raised(app, monkeypatch):
    def boom():
        raise RuntimeError("onnx corrupto")

    monkeypatch.setattr(face, "make_tracker", boom)
    win = MainWindow(Settings(detail=60, face_follow=True), camera_factory=Cam)
    win._tick()
    assert win.settings.face_follow is False and "onnx corrupto" in win.statusBar().currentMessage()
    win.close()


def test_checkbox_disabled_without_model(app, monkeypatch):
    from facia_app.ui import controls
    monkeypatch.setattr(controls, "model_path", lambda: None)
    win = MainWindow(Settings(), camera_factory=Cam)
    assert not win.panel.face.isEnabled() and "models/" in win.panel.face.toolTip()
    win.close()
