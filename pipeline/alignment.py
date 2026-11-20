"""Mise du modèle dans le repère du viewer du site.

Le viewer (components/tour/engine.ts) charge le .spz SANS transformation : la scène doit
déjà être en mètres, Y vers le haut, sol à y = 0 (yeux à ~1,6), centrée. Les scènes
existantes (appartement.spz) le confirment : sol à 0, médiane x/z à 0.

COLMAP donne une échelle et une orientation arbitraires. On estime :
  - la verticale : axes horizontaux des images (téléphone tenu droit), voir estimate_vertical,
    contrôlée indépendamment par le plan du sol du nuage de points (floor_plane_check) ;
  - le sol : un centile bas des hauteurs des points 3D ;
  - l'échelle : la hauteur médiane des caméras au-dessus du sol = camera_height_m ;
  - le cap : rotation autour de Y qui aligne les murs sur X/Z (boîte englobante minimale) ;
  - le centrage : médiane x/z des points à 0.
Transformation finale : X' = s · R · X + t.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .colmap_text import Images, Points, camera_centers_and_up, quat_to_rot


@dataclass
class Alignment:
    scale: float
    R: list[list[float]]
    t: list[float]
    up_consistency: float
    camera_height_model: float
    yaw_deg: float
    floor_model: float
    # Diagnostic (facultatif, pour state.json)
    camera_height_assumed_m: float = 1.5
    up_vector_colmap: list[float] | None = None
    vertical_angle_deg: float = 0.0       # angle entre la verticale estimée et l'axe -Y de COLMAP
    camera_roll_median_deg: float = 0.0   # roulis (axe horizontal de l'image vs horizon) : ~0 si tenu droit
    camera_roll_p90_deg: float = 0.0
    camera_pitch_median_deg: float = 0.0  # tangage : < 0 = regard vers le bas
    rotated_images: dict | None = None    # nb d'images par orientation détectée (0°, 90°, 180°, 270°)
    floor_check: dict | None = None       # contrôle indépendant : plan du sol dans le nuage de points
    overrides: dict | None = None

    def diagnostics(self) -> dict:
        return {
            "camera_height_assumed_m": self.camera_height_assumed_m,
            "scale_m_per_unit": round(self.scale, 6),
            "camera_height_model_units": round(self.camera_height_model, 6),
            "vertical_angle_deg": round(self.vertical_angle_deg, 2),
            "up_vector_colmap": [round(v, 4) for v in (self.up_vector_colmap or [])],
            "up_consistency": round(self.up_consistency, 4),
            "camera_roll_median_deg": round(self.camera_roll_median_deg, 2),
            "camera_roll_p90_deg": round(self.camera_roll_p90_deg, 2),
            "camera_pitch_median_deg": round(self.camera_pitch_median_deg, 2),
            "rotated_images": self.rotated_images or {},
            "yaw_deg": round(self.yaw_deg, 2),
            "floor_model_units": round(self.floor_model, 6),
            "floor_check": self.floor_check or {},
        }

    def matrix(self) -> np.ndarray:
        """Matrice 4x4 de la similitude."""
        m = np.eye(4)
        m[:3, :3] = self.scale * np.array(self.R)
        m[:3, 3] = self.t
        return m

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * pts @ np.array(self.R).T + np.array(self.t)

    def to_dict(self) -> dict:
        return asdict(self)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation minimale qui envoie le vecteur unitaire a sur b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposés : demi-tour autour d'un axe orthogonal à a
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 0, 1.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _trimmed_extent(xz: np.ndarray, lo: float = 5, hi: float = 95) -> np.ndarray:
    return np.percentile(xz, hi, axis=0) - np.percentile(xz, lo, axis=0)


def best_yaw(xz: np.ndarray) -> float:
    """Angle (degrés) autour de Y minimisant l'aire de la boîte englobante (5–95 %) au sol."""
    if len(xz) < 10:
        return 0.0
    best, best_area = 0.0, float("inf")
    for deg in np.arange(0, 90, 1.0):
        r = rot_y(deg)
        p = xz @ np.array([[r[0, 0], r[0, 2]], [r[2, 0], r[2, 2]]]).T
        ext = _trimmed_extent(p)
        area = float(ext[0] * ext[1])
        if area < best_area - 1e-12:
            best, best_area = float(deg), area
    # Convention des scènes du site : la plus grande dimension le long de Z.
    r = rot_y(best)
    p = xz @ np.array([[r[0, 0], r[0, 2]], [r[2, 0], r[2, 2]]]).T
    ext = _trimmed_extent(p)
    if ext[0] > ext[1]:
        best += 90.0
    return best


def compute_alignment(images: Images, points: Points, camera_height_m: float = 1.5,
                      floor_percentile: float = 2.0, align_walls: bool = True,
                      max_point_error: float = 2.0, min_track_len: int = 3) -> Alignment:
    if len(images.names) < 3:
        raise ValueError("pas assez d'images alignées pour estimer la verticale")
    centers, _ = camera_centers_and_up(images)
    v = estimate_vertical(images)
    up = v["up"]
    consistency = v["consistency"]
    R_up = rotation_between(up, np.array([0.0, 1.0, 0.0]))

    good = (points.error <= max_point_error) & (points.track_len >= min_track_len)
    pts = points.xyz[good] if good.sum() >= 50 else points.xyz
    if len(pts) < 10:
        raise ValueError("pas assez de points 3D pour estimer le sol")
    p1 = pts @ R_up.T
    c1 = centers @ R_up.T
    floor = float(np.percentile(p1[:, 1], floor_percentile))
    cam_h = float(np.median(c1[:, 1]) - floor)
    if cam_h <= 0:
        raise ValueError("caméras sous le sol estimé : verticale incohérente")
    s = camera_height_m / cam_h

    yaw = 0.0
    if align_walls:
        # Points entre 20 cm et ~2 m au-dessus du sol : murs et mobilier, pas sol ni plafond.
        h = (p1[:, 1] - floor) * s
        band = (h > 0.2) & (h < 2.0)
        xz = p1[band][:, [0, 2]] if band.sum() >= 50 else p1[:, [0, 2]]
        yaw = best_yaw(xz)
    R = rot_y(yaw) @ R_up

    p2 = s * pts @ R.T
    t = np.array([-float(np.median(p2[:, 0])), -s * floor, -float(np.median(p2[:, 2]))])
    vert = float(np.degrees(np.arccos(np.clip(np.dot(up, [0.0, -1.0, 0.0]), -1, 1))))
    floor_chk = floor_plane_check(s * pts @ R.T + t)
    return Alignment(scale=float(s), R=R.tolist(), t=t.tolist(), up_consistency=consistency,
                     camera_height_model=cam_h, yaw_deg=float(yaw), floor_model=floor,
                     camera_height_assumed_m=camera_height_m, up_vector_colmap=up.tolist(),
                     vertical_angle_deg=vert, camera_roll_median_deg=float(np.median(v["roll"])),
                     camera_roll_p90_deg=float(np.percentile(v["roll"], 90)),
                     camera_pitch_median_deg=float(np.median(v["pitch"])),
                     rotated_images=v["orientations"], floor_check=floor_chk)


FLOOR_TILT_TARGET_DEG = 3.0
FLOOR_TILT_WARN_DEG = 5.0


def floor_plane_check(aligned: np.ndarray, window: tuple[float, float] = (-0.3, 0.5),
                      max_normal_deg: float = 30.0, threshold_m: float = 0.03,
                      iterations: int = 1000, seed: int = 0) -> dict:
    """Contrôle INDÉPENDANT de la verticale (qui vient des caméras) : plus grand plan quasi
    horizontal (normale à moins de 30° de Y) parmi les points du bas de la scène alignée
    (y entre -0,3 et 0,5 m), trouvé par RANSAC puis réajusté (ACP des inliers).
    Son inclinaison doit être < 3° ; au-delà de 5°, la verticale ou le sol est suspect."""
    lo, hi = window
    cand = aligned[(aligned[:, 1] > lo) & (aligned[:, 1] < hi)]
    if len(cand) < 30:
        return {"status": "indéterminé", "reason": f"{len(cand)} points dans la bande du sol"}
    rng = np.random.default_rng(seed)
    cos_max = math.cos(math.radians(max_normal_deg))
    best_mask, best_n = None, 0
    for _ in range(iterations):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12 or abs(n[1]) / norm < cos_max:
            continue
        n /= norm
        mask = np.abs((cand - a) @ n) < threshold_m
        k = int(mask.sum())
        if k > best_n:
            best_mask, best_n = mask, k
    if best_mask is None or best_n < 20:
        return {"status": "indéterminé", "reason": "aucun plan horizontal trouvé"}
    inl = cand[best_mask]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[2] * (1 if vt[2][1] >= 0 else -1)
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    status = "ok" if tilt < FLOOR_TILT_TARGET_DEG else ("à surveiller" if tilt <= FLOOR_TILT_WARN_DEG
                                                        else "AVERTISSEMENT")
    return {"tilt_deg": round(tilt, 2), "status": status, "target_deg": FLOOR_TILT_TARGET_DEG,
            "warn_deg": FLOOR_TILT_WARN_DEG, "plane_height_m": round(float(centroid[1]), 3),
            "normal": [round(float(x), 4) for x in n], "inliers": best_n, "window_points": len(cand),
            "tilt_towards_deg": round(float(np.degrees(np.arctan2(n[0], n[2]))), 1)}


def estimate_vertical(images: Images, max_iter: int = 20) -> dict:
    """Verticale du monde à partir des caméras, robuste au tangage et aux images tournées de 90°.

    Hypothèse (celle de « model_orientation_aligner IMAGE-ORIENTATION » de COLMAP) : l'appareil est
    tenu droit, donc l'axe HORIZONTAL de l'image reste horizontal dans le monde, quel que soit le
    tangage (filmer le sol ou le plafond). La verticale est la direction la plus orthogonale à ces
    axes (plus petite valeur propre de Σ x xᵀ), son sens donné par les « haut » des images.
    Chaque image peut être en paysage ou en portrait (0°, 90°, 180°, 270°) : on lui attribue
    l'orientation la plus cohérente avec la verticale courante, et on itère.
    Mesuré sur playroom (photos d'appareil, 72/217 en portrait) : roulis médian 1,8° contre 6,6°
    avec la simple moyenne des « haut », et verticale corrigée de 5,8°.
    """
    R = quat_to_rot(images.qvec)
    Rt = np.transpose(R, (0, 2, 1))                                    # caméra -> monde
    ex, ey = Rt @ np.array([1.0, 0, 0]), Rt @ np.array([0, 1.0, 0])
    ups = np.stack([-ey, ex, ey, -ex], axis=1)                         # haut de l'image : 0°, 90°, 180°, 270°
    rights = np.stack([ex, ey, -ex, -ey], axis=1)                      # droite correspondante
    g = (-ey).sum(axis=0)
    g /= np.linalg.norm(g)
    idx = np.arange(len(ex))
    for _ in range(max_iter):
        k = np.argmax(ups @ g, axis=1)
        up_a, right_a = ups[idx, k], rights[idx, k]
        _, vecs = np.linalg.eigh(right_a.T @ right_a)
        g_new = vecs[:, 0] * np.sign(vecs[:, 0] @ up_a.sum(axis=0) or 1.0)
        converged = float(g_new @ g) > np.cos(np.radians(0.01))
        g = g_new
        if converged:
            break
    k = np.argmax(ups @ g, axis=1)
    up_a, right_a = ups[idx, k], rights[idx, k]
    roll = np.degrees(np.arcsin(np.clip(np.abs(right_a @ g), 0, 1)))
    pitch = np.degrees(np.arcsin(np.clip(np.cross(right_a, up_a) @ g, -1, 1)))
    counts = np.bincount(k, minlength=4)
    return {"up": g, "roll": roll, "pitch": pitch,
            "consistency": float(np.linalg.norm(up_a.sum(axis=0)) / len(up_a)),
            "orientations": {"0": int(counts[0]), "90": int(counts[1]), "180": int(counts[2]),
                             "270": int(counts[3])}}


def apply_overrides(al: Alignment, scale_factor: float | None = None, yaw_deg: float | None = None,
                    floor_offset_m: float | None = None) -> Alignment:
    """Corrections manuelles appliquées APRÈS l'estimation automatique, dans le repère aligné :
    X'' = f · Ry(θ) · X' + (0, dy, 0). La scène étant centrée et le sol à 0, l'échelle et la
    rotation autour de l'origine gardent le sol à 0 et le centre en place."""
    f = 1.0 if scale_factor is None else float(scale_factor)
    th = 0.0 if yaw_deg is None else float(yaw_deg)
    dy = 0.0 if floor_offset_m is None else float(floor_offset_m)
    if f <= 0:
        raise ValueError("alignment.override.scale_factor doit être > 0")
    Ry = rot_y(th)
    R = Ry @ np.array(al.R)
    t = f * Ry @ np.array(al.t) + np.array([0.0, dy, 0.0])
    data = al.to_dict()
    data.update(scale=al.scale * f, R=R.tolist(), t=t.tolist(), yaw_deg=al.yaw_deg + th,
                overrides={"scale_factor": f, "yaw_deg": th, "floor_offset_m": dy})
    return Alignment(**data)


# ── Traduction en actions splat-transform ───────────────────────────────────
#
# Mesuré sur splat-transform v3.12 : les actions -r/-t opèrent dans un repère interne tourné
# de 180° autour de Z par rapport aux coordonnées du fichier (D = diag(-1,-1,1)). Donc, en
# coordonnées fichier :  -r a,0,0 = Rx(-a) ; -r 0,b,0 = Ry(-b) ; -r 0,0,c = Rz(c) ;
# -t x,y,z = translation (-x,-y,z) ; -s k = échelle k. Les actions s'appliquent dans l'ordre.
# tests/test_alignment.py vérifie cette traduction contre splat-transform s'il est installé.

def euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """Décompose R = Rz(g) · Ry(b) · Rx(a), angles en degrés (a, b, g)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    b = math.asin(sy)
    if abs(sy) < 1 - 1e-9:
        a = math.atan2(R[2, 1], R[2, 2])
        g = math.atan2(R[1, 0], R[0, 0])
    else:  # blocage de cardan
        a = 0.0
        g = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(a), math.degrees(b), math.degrees(g)


def _num(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def splat_transform_args(al: Alignment) -> list[str]:
    a, b, g = euler_zyx(np.array(al.R))
    tx, ty, tz = al.t
    return [
        "-r", f"{_num(-a)},0,0",
        "-r", f"0,{_num(-b)},0",
        "-r", f"0,0,{_num(g)}",
        "-s", _num(al.scale),
        "-t", f"{_num(-tx)},{_num(-ty)},{_num(tz)}",
    ]
