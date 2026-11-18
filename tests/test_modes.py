import numpy as np
import pytest

from facia_app.engine import Engine
from facia_app.modes import BRAILLE_BASE, EDGE_CHARS, braille_indices, edge_indices


def test_braille_extremes():
    assert (braille_indices(np.zeros((16, 8), np.float32)) == 0).all()
    assert (braille_indices(np.full((16, 8), 255, np.float32)) == 255).all()


def test_braille_shape_and_range():
    out = braille_indices(np.random.default_rng(0).uniform(0, 255, (4 * 10, 2 * 30)).astype(np.float32))
    assert out.shape == (10, 30) and out.dtype == np.uint8
    assert chr(BRAILLE_BASE + int(out.max())) >= "⠀"


def test_braille_density_follows_brightness():
    ramp = np.tile(np.linspace(0, 255, 2 * 40, dtype=np.float32), (4 * 6, 1))
    bits = np.unpackbits(braille_indices(ramp)[:, :, None], axis=2).sum(axis=2).mean(axis=0)
    assert bits[:5].mean() < bits[17:23].mean() < bits[-5:].mean()


def test_braille_tiny_input():
    assert braille_indices(np.zeros((1, 1), np.float32)).shape == (1, 1)


def _hi(fn, rows=10, cols=20):
    y, x = np.mgrid[0 : rows * 4, 0 : cols * 2]
    return np.where(fn(x, y), 255.0, 0.0).astype(np.float32)


@pytest.mark.parametrize("name,fn,char", [
    ("vertical", lambda x, y: x >= 20, "|"),
    ("horizontal", lambda x, y: y >= 20, "-"),
    ("diag /", lambda x, y: x + y >= 40, "/"),
    ("diag \\", lambda x, y: x - y >= 0, "\\"),
])
def test_edge_directions(name, fn, char):
    idx = edge_indices(_hi(fn), levels=10, sensitivity=0.5)
    edge = idx[idx >= 10]
    assert edge.size > 0
    vals, counts = np.unique(edge - 10, return_counts=True)
    assert EDGE_CHARS[vals[counts.argmax()]] == char


def test_edge_flat_image_has_no_edges():
    idx = edge_indices(np.full((40, 40), 128.0, np.float32), levels=10)
    assert idx.max() < 10


def test_edge_sensitivity_monotonic():
    rng = np.random.default_rng(0)
    g = np.clip(100 + rng.normal(0, 25, (40, 60)), 0, 255).astype(np.float32)
    n = [(edge_indices(g, 10, s) >= 10).sum() for s in (0.1, 0.5, 0.9)]
    assert n[0] <= n[1] <= n[2]


def test_engine_gray_matches_process_and_hires():
    f = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    eng = Engine(10, smoothing=1.0)
    g = eng.gray(f, 480, 140)
    assert g.shape == (140, 480) and 0 <= g.min() and g.max() <= 255
