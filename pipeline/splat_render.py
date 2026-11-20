"""Rendu CPU (numpy) de gaussiennes 3D pour un contrôle visuel sans GPU.

Couleur = harmonique de degré 0 seulement (pas d'effet de vue) ; projection EWA et
compositing avant -> arrière comme 3DGS. Lent (~10 s par vue 320x240 pour 500 000 gaussiennes)
mais indépendant de gsplat et de splat-transform : sert au diagnostic et à la planche de vues
de l'étape 5. Caméras au format OpenCV (x droite, y bas, z avant), matrice monde -> caméra 4x4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .plyutil import read_header, read_vertex_columns

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    xyz: np.ndarray       # (N, 3)
    cov: np.ndarray       # (N, 3, 3)
    opacity: np.ndarray   # (N,) 0..1
    rgb: np.ndarray       # (N, 3) 0..1 (degré 0)
    log_scale: np.ndarray  # (N, 3)

    def __len__(self) -> int:
        return len(self.xyz)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_gaussians(path: Path) -> Gaussians:
    cols = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2",
            "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    d = read_vertex_columns(Path(path), cols)
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    q = np.stack([d[f"rot_{i}"] for i in range(4)], 1)
    ls = np.stack([d[f"scale_{i}"] for i in range(3)], 1)
    M = quat_to_rot(q) * np.exp(ls)[:, None, :]
    rgb = np.clip(0.5 + SH_C0 * np.stack([d[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    return Gaussians(xyz, M @ M.transpose(0, 2, 1), 1 / (1 + np.exp(-d["opacity"])), rgb, ls)


def splat_stats(path: Path) -> dict:
    """Distributions utiles au diagnostic (centiles 1, 10, 50, 90, 99)."""
    g = load_gaussians(path)
    ps = [1, 10, 50, 90, 99]

    def q(x):
        return [round(float(v), 5) for v in np.percentile(x, ps)]

    smax = np.exp(g.log_scale).max(1)
    center = np.median(g.xyz, 0)
    lo, hi = np.percentile(g.xyz, 0.5, 0) - 0.3, np.percentile(g.xyz, 99.5, 0) + 0.3
    outside = ~np.all((g.xyz >= lo) & (g.xyz <= hi), 1)
    return {
        "file": str(path), "count": len(g), "properties": len(read_header(Path(path)).props()),
        "percentiles": ps,
        "position": {a: q(g.xyz[:, i]) for i, a in enumerate("xyz")},
        "scale_max_axis_m": q(smax), "log_scale": q(g.log_scale.ravel()),
        "opacity": q(g.opacity), "rgb_dc": {c: q(g.rgb[:, i]) for i, c in enumerate("rgb")},
        "radius_from_median_m": q(np.linalg.norm(g.xyz - center, axis=1)),
        "outside_box": {"count": int(outside.sum()), "fraction": round(float(outside.mean()), 4),
                        "scale_max_axis_median_m": round(float(np.median(smax[outside])), 4) if outside.any() else None},
    }


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Monde -> caméra (OpenCV) pour une caméra en eye regardant target."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(f, right), f])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def render(g: Gaussians, w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float, W: int, H: int,
           bg: float = 0.0, max_pairs: int = 30_000_000) -> np.ndarray:
    """Image (H, W, 3) float 0..1."""
    Rw, tw = w2c[:3, :3], w2c[:3, 3]
    pc = g.xyz @ Rw.T + tw
    z = pc[:, 2]
    keep = z > 0.05
    zs = np.where(keep, z, 1.0)
    u = fx * pc[:, 0] / zs + cx
    v = fy * pc[:, 1] / zs + cy
    keep &= (u > -W) & (u < 2 * W) & (v > -H) & (v < 2 * H)
    idx = np.nonzero(keep)[0]
    pc, z, u, v = pc[idx], z[idx], u[idx], v[idx]
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * pc[:, 0] / z ** 2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * pc[:, 1] / z ** 2
    T = J @ Rw
    c2 = T @ g.cov[idx] @ T.transpose(0, 2, 1)
    a, b, c = c2[:, 0, 0] + 0.3, c2[:, 0, 1], c2[:, 1, 1] + 0.3   # filtre passe-bas de 3DGS
    det = a * c - b * b
    ia, ib, ic = c / det, -b / det, a / det
    lam = 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    r = np.ceil(3 * np.sqrt(lam))
    x0 = np.clip(np.floor(u - r), 0, W - 1).astype(np.int64)
    x1 = np.clip(np.ceil(u + r), 0, W - 1).astype(np.int64)
    y0 = np.clip(np.floor(v - r), 0, H - 1).astype(np.int64)
    y1 = np.clip(np.ceil(v + r), 0, H - 1).astype(np.int64)
    ok = (u + r >= 0) & (u - r < W) & (v + r >= 0) & (v - r < H)
    sel = np.nonzero(ok)[0]
    sel = sel[np.argsort(z[sel], kind="stable")]           # avant -> arrière
    nx = x1[sel] - x0[sel] + 1
    area = nx * (y1[sel] - y0[sel] + 1)
    starts = np.concatenate([[0], np.cumsum(area)])
    img = np.zeros((H * W, 3))
    trans = np.ones(H * W)
    lo = 0
    while lo < len(sel):                                   # lots de gaussiennes déjà triées
        hi = max(int(np.searchsorted(starts, starts[lo] + max_pairs, side="right")) - 1, lo + 1)
        gs, ar, nxg = sel[lo:hi], area[lo:hi], nx[lo:hi]
        rep = np.repeat(np.arange(len(gs)), ar)
        off = np.arange(int(ar.sum())) - np.repeat(np.cumsum(ar) - ar, ar)
        px = x0[gs][rep] + off % nxg[rep]
        py = y0[gs][rep] + off // nxg[rep]
        gi = gs[rep]
        dx, dy = px + 0.5 - u[gi], py + 0.5 - v[gi]
        power = -0.5 * (ia[gi] * dx * dx + 2 * ib[gi] * dx * dy + ic[gi] * dy * dy)
        alpha = np.minimum(0.99, g.opacity[idx[gi]] * np.exp(np.minimum(power, 0)))
        m = (alpha >= 1 / 255) & (power <= 0)
        pix, alpha, col = (py * W + px)[m], alpha[m], g.rgb[idx[gi[m]]]
        o = np.argsort(pix, kind="stable")                 # par pixel, profondeur conservée
        pix, alpha, col = pix[o], alpha[o], col[o]
        la = np.log1p(-alpha)
        cs = np.cumsum(la)
        first = np.r_[True, pix[1:] != pix[:-1]]
        seg = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))
        base = np.where(seg > 0, cs[seg - 1], 0.0)
        w = trans[pix] * np.exp(cs - la - base) * alpha
        np.add.at(img, pix, w[:, None] * col)
        last = np.r_[pix[1:] != pix[:-1], True]
        trans[pix[last]] *= np.exp(cs[last] - base[last])
        lo = hi
    img += trans[:, None] * bg
    return np.clip(img.reshape(H, W, 3), 0, 1)


def room_sheet(g: Gaussians, eye, W: int = 320, H: int = 240, fov_deg: float = 70.0,
               pitch: float = -0.15) -> np.ndarray:
    """Planche 2x2 (uint8) : 4 vues depuis eye (repère du site, y en haut), caps 0/90/180/270°."""
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    eye = np.asarray(eye, float)
    tiles = []
    for yaw in (0, 90, 180, 270):
        d = np.array([np.sin(np.radians(yaw)), pitch, np.cos(np.radians(yaw))])
        tiles.append((render(g, look_at(eye, eye + d), f, f, W / 2, H / 2, W, H) * 255).astype(np.uint8))
    return np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
