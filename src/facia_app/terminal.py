"""Renderer de consola."""

import shutil
import sys
import time

import numpy as np

from .capture import CameraSource
from .config import Settings
from .engine import Engine, fit_grid


def run(settings: Settings) -> None:
    palette = np.frombuffer(settings.palette.encode("ascii"), dtype=np.uint8)
    engine = Engine(len(palette), settings.gamma, settings.smoothing)
    cam = CameraSource(settings.camera_index)
    cam.start()
    out, last_seq = sys.stdout, 0
    out.write("\033[2J\033[?25l")
    try:
        while True:
            seq, frame = cam.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            if settings.mirror:
                frame = frame[:, ::-1]
            cols, rows = shutil.get_terminal_size((120, 40))
            cols, rows = fit_grid(frame.shape[:2], cols, rows - 1, settings.char_aspect)
            idx = engine.process(np.ascontiguousarray(frame), cols, rows)
            chars = palette[idx]
            chars = np.hstack([chars, np.full((rows, 1), 10, np.uint8)])
            out.write("\033[H" + chars.tobytes().decode("ascii") + "\033[J")
            out.flush()
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\033[?25h\n")
        cam.stop()
