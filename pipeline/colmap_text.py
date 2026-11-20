"""Lecture des modèles COLMAP au format texte (images.txt, points3D.txt).

Format (doc/format.rst de COLMAP 4.2.1) :
  images.txt   : IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME  puis une ligne POINTS2D[]
  points3D.txt : POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
La pose (q, t) est world -> caméra, convention OpenCV (x droite, y bas, z avant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Images:
    names: list[str]
    qvec: np.ndarray  # (N, 4) qw, qx, qy, qz
    tvec: np.ndarray  # (N, 3)
    n_points3d: np.ndarray | None = field(default=None)  # (N,) observations reliées à un point 3D

    def subset(self, mask: np.ndarray) -> "Images":
        mask = np.asarray(mask, dtype=bool)
        return Images([n for n, k in zip(self.names, mask) if k], self.qvec[mask], self.tvec[mask],
                      None if self.n_points3d is None else self.n_points3d[mask])


@dataclass
class Points:
    xyz: np.ndarray        # (M, 3)
    error: np.ndarray      # (M,)
    track_len: np.ndarray  # (M,)


def _data_lines(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                yield s
            elif not s and not line.startswith("#"):
                # Ligne POINTS2D vide (image sans point) : il faut la conserver pour l'alternance.
                yield ""


def read_images(path: Path) -> Images:
    names, q, t, n3d = [], [], [], []
    lines = list(_data_lines(path))
    # Alternance ligne image / ligne points 2D. On ignore les lignes vides finales.
    i = 0
    while i < len(lines):
        s = lines[i]
        if not s:
            i += 1
            continue
        parts = s.split()
        if len(parts) < 10:
            raise ValueError(f"{path}: ligne image inattendue : {s[:80]}")
        qi = [float(x) for x in parts[1:5]]
        ti = [float(x) for x in parts[5:8]]
        pts2d = lines[i + 1].split() if i + 1 < len(lines) else []
        i += 2  # ligne POINTS2D (éventuellement vide) : X, Y, POINT3D_ID (-1 = aucun)
        # Une image non recalée éventuelle porte une pose identité : on l'ignore.
        if qi == [1.0, 0.0, 0.0, 0.0] and ti == [0.0, 0.0, 0.0]:
            continue
        q.append(qi)
        t.append(ti)
        names.append(" ".join(parts[9:]))
        n3d.append(sum(x != "-1" for x in pts2d[2::3]))
    return Images(names, np.array(q, dtype=np.float64).reshape(-1, 4),
                  np.array(t, dtype=np.float64).reshape(-1, 3), np.array(n3d, dtype=np.int64))


def usable_mask(images: Images, min_points3d: int = 15, max_distance_ratio: float = 10.0) -> np.ndarray:
    """Images réellement exploitables d'un modèle COLMAP.

    Le global_mapper de COLMAP 4 garde « recalées » des images dont toutes les observations ont
    été filtrées : 0 point 3D et une pose absurde (sur playroom : 13 images, jusqu'à 200 000 fois
    la distance médiane). Elles faussent le taux d'alignement et l'échelle de scène de gsplat
    (fondée sur la caméra la plus éloignée), ce qui a fait diverger l'entraînement.
    On garde les images avec >= min_points3d points 3D ET à moins de max_distance_ratio fois la
    distance médiane des caméras à leur centre (médian)."""
    n = len(images.names)
    if n == 0:
        return np.zeros(0, dtype=bool)
    ok = np.ones(n, dtype=bool) if images.n_points3d is None else images.n_points3d >= min_points3d
    centers, _ = camera_centers_and_up(images)
    ref = centers[ok] if ok.sum() >= 3 else centers
    med = np.median(ref, axis=0)
    dist = np.linalg.norm(centers - med, axis=1)
    md = float(np.median(np.linalg.norm(ref - med, axis=1)))
    if md > 0:
        ok &= dist <= max_distance_ratio * md
    return ok


def read_points(path: Path) -> Points:
    xyz, err, tl = [], [], []
    for s in _data_lines(path):
        if not s:
            continue
        parts = s.split()
        xyz.append([float(x) for x in parts[1:4]])
        err.append(float(parts[7]))
        tl.append((len(parts) - 8) // 2)
    return Points(np.array(xyz, dtype=np.float64).reshape(-1, 3),
                  np.array(err, dtype=np.float64), np.array(tl, dtype=np.int64))


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) qw,qx,qy,qz -> (N,3,3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def camera_centers_and_up(images: Images) -> tuple[np.ndarray, np.ndarray]:
    """Centres optiques (N,3) et vecteur « haut » de chaque caméra en coordonnées monde (N,3)."""
    R = quat_to_rot(images.qvec)                       # world -> cam
    Rt = np.transpose(R, (0, 2, 1))                    # cam -> world
    centers = -np.einsum("nij,nj->ni", Rt, images.tvec)
    up = Rt @ np.array([0.0, -1.0, 0.0])               # -y caméra (OpenCV) = haut de l'image
    return centers, up


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) -> qw,qx,qy,qz (qw >= 0), méthode de Shepperd (stable)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(max(1e-300, 1 + R[i, i] - R[j, j] - R[k, k]))
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q * (1 if q[0] >= 0 else -1) / np.linalg.norm(q)


def transform_model_txt(src: Path, dst: Path, s: float, R: np.ndarray, t: np.ndarray,
                        keep: set[str] | None = None) -> None:
    """Réécrit un modèle TXT (cameras, images, points3D) dans le repère X' = s·R·X + t.

    Pose monde -> caméra : R_c' = R_c·Rᵀ, t_c' = s·t_c − R_c'·t (les coordonnées caméra sont
    multipliées par s ; les intrinsèques et les observations 2D ne changent pas). keep : noms
    des images à garder (les autres sont retirées, ainsi que leurs entrées de TRACK). rigs.txt et
    frames.txt (COLMAP 4) ne sont PAS recopiés : seul gsplat lit ce modèle, et son lecteur
    n'utilise que cameras/images/points3D.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "cameras.txt").write_text((src / "cameras.txt").read_text(encoding="utf-8"), encoding="utf-8")

    out, expect_pose, dropped, skip_next = [], True, set(), False
    for line in (src / "images.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            out.append(line)
            continue
        if skip_next:
            skip_next, expect_pose = False, True
            continue
        if expect_pose and line.strip():
            p = line.split()
            if keep is not None and " ".join(p[9:]) not in keep:
                dropped.add(p[0])
                skip_next = True
                continue
            q, tc = np.array([float(x) for x in p[1:5]]), np.array([float(x) for x in p[5:8]])
            Rc = quat_to_rot(q[None])[0] @ R.T
            tn = s * tc - Rc @ t
            qn = _rot_to_quat(Rc)
            out.append(" ".join([p[0], *(f"{v:.12g}" for v in qn), *(f"{v:.12g}" for v in tn), *p[8:]]))
            expect_pose = False
        else:
            out.append(line)          # ligne POINTS2D (éventuellement vide)
            expect_pose = True
    (dst / "images.txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    out = []
    for line in (src / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        p = line.split()
        x = s * (R @ np.array([float(v) for v in p[1:4]])) + t
        track = p[8:]
        if dropped:
            track = [v for k in range(0, len(track), 2) if track[k] not in dropped for v in track[k:k + 2]]
            if not track:
                continue          # point vu seulement par des images retirées
        out.append(" ".join([p[0], *(f"{v:.9g}" for v in x), *p[4:8], *track]))
    (dst / "points3D.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


_CAMERA_MODELS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4,
                  "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6}


def write_bin_for_gsplat(txt_dir: Path, dst: Path) -> None:
    """Modèle TXT -> cameras/images/points3D.bin au format binaire CLASSIQUE (celui de COLMAP 3).

    Pourquoi ne pas utiliser « colmap model_converter » : en COLMAP 4, les poses viennent de
    frames.txt (images.txt n'est plus lu pour la pose) ; et le lecteur TXT du pycolmap de gsplat
    est cassé sous Python 3 (np.array(map(...))). Le lecteur BIN de ce pycolmap, lui, fonctionne.
    """
    import struct

    dst.mkdir(parents=True, exist_ok=True)
    cams = [s.split() for s in _data_lines(txt_dir / "cameras.txt") if s]
    with open(dst / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for p in cams:
            params = [float(x) for x in p[4:]]
            f.write(struct.pack("<iiQQ", int(p[0]), _CAMERA_MODELS[p[1]], int(p[2]), int(p[3])))
            f.write(struct.pack(f"<{len(params)}d", *params))

    lines = [s for s in (txt_dir / "images.txt").read_text(encoding="utf-8").splitlines() if not s.startswith("#")]
    recs = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        head = lines[i].split()
        pts = lines[i + 1].split() if i + 1 < len(lines) else []
        recs.append((head, pts))
        i += 2
    with open(dst / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(recs)))
        for head, pts in recs:
            f.write(struct.pack("<i7di", int(head[0]), *(float(x) for x in head[1:8]), int(head[8])))
            f.write(" ".join(head[9:]).encode() + b"\0")
            n = len(pts) // 3
            f.write(struct.pack("<Q", n))
            for j in range(n):
                f.write(struct.pack("<ddq", float(pts[3 * j]), float(pts[3 * j + 1]), int(pts[3 * j + 2])))

    plines = [s.split() for s in _data_lines(txt_dir / "points3D.txt") if s]
    with open(dst / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(plines)))
        for p in plines:
            track = [int(x) for x in p[8:]]
            f.write(struct.pack("<Q3d3Bd", int(p[0]), *(float(x) for x in p[1:4]),
                                *(int(x) for x in p[4:7]), float(p[7])))
            f.write(struct.pack("<Q", len(track) // 2))
            f.write(struct.pack(f"<{len(track)}i", *track))


def read_cameras(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    out = {}
    for s in _data_lines(path):
        if s:
            p = s.split()
            out[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    return out


def compare_models_txt(expected: Path, actual: Path, tol: float = 1e-6) -> list[str]:
    """Écarts entre deux modèles TXT (caméras, poses, points). Liste vide = identiques."""
    problems = []
    ce, ca = read_cameras(expected / "cameras.txt"), read_cameras(actual / "cameras.txt")
    if ce.keys() != ca.keys():
        problems.append(f"caméras : ids {sorted(ce)} != {sorted(ca)}")
    for cid in ce.keys() & ca.keys():
        (me, we, he, pe), (ma, wa, ha, pa) = ce[cid], ca[cid]
        if (me, we, he) != (ma, wa, ha) or len(pe) != len(pa) or not np.allclose(pe, pa, rtol=tol, atol=tol):
            problems.append(f"caméra {cid} : {ce[cid]} != {ca[cid]}")
    ie, ia = read_images(expected / "images.txt"), read_images(actual / "images.txt")
    if sorted(ie.names) != sorted(ia.names):
        problems.append(f"images : {len(ie.names)} écrites, {len(ia.names)} relues")
    else:
        order = {n: k for k, n in enumerate(ia.names)}
        idx = np.array([order[n] for n in ie.names])
        dq = np.minimum(np.abs(ie.qvec - ia.qvec[idx]).max(1), np.abs(ie.qvec + ia.qvec[idx]).max(1))
        dt = np.abs(ie.tvec - ia.tvec[idx]).max(1) / np.maximum(1.0, np.abs(ie.tvec).max(1))
        if dq.max() > tol or dt.max() > tol:
            problems.append(f"poses : écart max quaternion {dq.max():.2e}, translation {dt.max():.2e}")
        if not np.array_equal(ie.n_points3d, ia.n_points3d[idx]):
            problems.append("observations 2D-3D différentes")
    pe, pa = read_points(expected / "points3D.txt"), read_points(actual / "points3D.txt")
    if len(pe.xyz) != len(pa.xyz):
        problems.append(f"points 3D : {len(pe.xyz)} écrits, {len(pa.xyz)} relus")
    elif len(pe.xyz):
        d = np.abs(np.sort(pe.xyz, axis=0) - np.sort(pa.xyz, axis=0)).max()
        if d > tol * max(1.0, np.abs(pe.xyz).max()):
            problems.append(f"points 3D : écart max {d:.2e}")
    return problems
