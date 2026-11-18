import time

import numpy as np

from facia_app import capture


class FlakyCap:
    def __init__(self, fail_at=(8,)):
        self.i, self.fail_at, self.released_in = 0, set(fail_at), None

    def isOpened(self):
        return True

    def read(self):
        self.i += 1
        if self.i in self.fail_at:
            return False, None
        time.sleep(0.002)
        return True, np.zeros((10, 10, 3), np.uint8)

    def release(self):
        import threading
        self.released_in = threading.current_thread().name


def test_transient_failure_does_not_stop(monkeypatch):
    cap = FlakyCap()
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda s: cap)
    cam = capture.CameraSource(0)
    cam.start()
    time.sleep(0.4)
    assert cam.state is capture.State.LIVE
    cam.stop()


def test_release_called_from_capture_thread(monkeypatch):
    cap = FlakyCap(fail_at=())
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda s: cap)
    cam = capture.CameraSource(0)
    cam.start()
    time.sleep(0.2)
    cam.stop()
    assert cap.released_in == cam._thread.name
