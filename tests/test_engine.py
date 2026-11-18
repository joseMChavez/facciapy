import numpy as np
import pytest

from facia_app.engine import Engine, fit_grid


def frame(h=720, w=1280):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("cols,rows", [(120, 40), (80, 1), (80, 0), (1, 1), (400, 120)])
def test_sizes_never_fail(cols, rows):
    out = Engine(10).process(frame(), cols, rows)
    assert out.shape == (max(1, rows), max(1, cols))
    assert out.max() < 10


def test_fit_grid_respects_limits():
    cols, rows = fit_grid((720, 1280), 120, 10)
    assert rows <= 10 and cols >= 1


def test_gradient_is_monotonic():
    g = np.tile(np.linspace(0, 255, 1280, dtype=np.uint8), (720, 1))
    bgr = np.dstack([g, g, g])
    row = Engine(10, smoothing=1.0).process(bgr, 40, 10)[5]
    assert (np.diff(row.astype(int)) >= 0).all()


def test_static_scene_low_flicker():
    rng = np.random.default_rng(1)
    base = np.full((720, 1280, 3), 120, np.uint8)
    base[:, :640] = 60
    eng, prev, rates = Engine(10), None, []
    for i in range(40):
        f = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        idx = eng.process(f, 120, 33)
        if prev is not None and i > 10:
            rates.append((idx != prev).mean() * 100)
        prev = idx
    assert np.mean(rates) < 5
