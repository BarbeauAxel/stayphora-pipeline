import numpy as np

from pipeline.splat_render import Gaussians, look_at, render


def _g(xyz, rgb, sigma=0.05, opacity=0.99):
    n = len(xyz)
    cov = np.repeat((np.eye(3) * sigma ** 2)[None], n, 0)
    return Gaussians(np.asarray(xyz, float), cov, np.full(n, opacity), np.asarray(rgb, float),
                     np.full((n, 3), np.log(sigma)))


def test_projection_and_color():
    # caméra en (0,0,0) regardant +z, y en haut (repère direct) : sa droite est -x.
    # Gaussienne rouge en (-0.5, 0, 2) -> à droite du centre, u = 100 * 0.5 / 2 + 50 = 75.
    g = _g([[-0.5, 0, 2]], [[1, 0, 0]])
    w2c = look_at((0, 0, 0), (0, 0, 1))
    img = render(g, w2c, 100, 100, 50, 50, 100, 100)
    y, x = np.unravel_index(img[..., 0].argmax(), img.shape[:2])
    assert abs(x - 74.5) <= 1 and abs(y - 49.5) <= 1   # centres de pixel à +0,5
    assert img[y, x, 0] > 0.9 and img[y, x, 1] < 0.05


def test_front_occludes_back():
    g = _g([[0, 0, 3], [0, 0, 1.5]], [[0, 0, 1], [0, 1, 0]], sigma=0.2)
    img = render(g, look_at((0, 0, 0), (0, 0, 1)), 100, 100, 50, 50, 100, 100)
    assert img[50, 50, 1] > 0.9 and img[50, 50, 2] < 0.05


def test_up_is_up():
    # monde y en haut : un point au-dessus de l'axe de visée apparaît en haut de l'image
    g = _g([[0, 1, 3]], [[1, 1, 1]])
    img = render(g, look_at((0, 0, 0), (0, 0, 1)), 100, 100, 50, 50, 100, 100)
    y, _ = np.unravel_index(img[..., 0].argmax(), img.shape[:2])
    assert y < 50
