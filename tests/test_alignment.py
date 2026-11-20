"""Alignement COLMAP -> repère du viewer, et traduction en actions splat-transform."""

import numpy as np
import pytest

from pipeline.alignment import (Alignment, compute_alignment, euler_zyx, rot_x, rot_y, rot_z,
                                splat_transform_args)
from pipeline.colmap_text import Images, Points
from pipeline.plyutil import read_vertex_columns
from tests.conftest import run_st, write_gaussian_ply


def _rot_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def synthetic_scene(seed=0):
    """Pièce de 6 m (z) x 4 m (x) x 2,5 m, caméras à 1,5 m, dans le repère du site."""
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(-2, 2, 3000), np.zeros(3000), rng.uniform(-3, 3, 3000)]
    walls = []
    for x in (-2, 2):
        walls.append(np.c_[np.full(1500, x), rng.uniform(0, 2.5, 1500), rng.uniform(-3, 3, 1500)])
    for z in (-3, 3):
        walls.append(np.c_[rng.uniform(-2, 2, 1000), rng.uniform(0, 2.5, 1000), np.full(1000, z)])
    pts = np.vstack([floor, *walls])
    # Caméras : chemin dans la pièce, regard horizontal (cap aléatoire), léger tangage.
    n = 120
    centers = np.c_[rng.uniform(-1.2, 1.2, n), 1.5 + rng.normal(0, 0.05, n), rng.uniform(-2, 2, n)]
    qs, ts = [], []
    flip = np.diag([1.0, -1.0, -1.0])  # caméra OpenCV (y bas, z avant) vs monde y haut
    for c in centers:
        cam_to_world = rot_y(rng.uniform(0, 360)) @ rot_x(rng.normal(0, 5)) @ flip
        R = cam_to_world.T                     # world -> cam
        qs.append(_rot_to_quat(R))
        ts.append(-R @ c)
    return pts, centers, np.array(qs), np.array(ts)


def to_colmap_frame(pts, qs, ts, s, R, t):
    """Applique une similitude arbitraire X_c = s R X + t (repère « COLMAP »)."""
    from pipeline.colmap_text import quat_to_rot
    pts_c = s * pts @ R.T + t
    Rw = quat_to_rot(qs)
    q2, t2 = [], []
    for Ri, ti in zip(Rw, ts):
        Rn = Ri @ R.T
        q2.append(_rot_to_quat(Rn))
        t2.append(s * ti - Rn @ t)
    return pts_c, np.array(q2), np.array(t2)


def test_alignement_retrouve_le_repere_du_site():
    pts, centers, qs, ts = synthetic_scene()
    R_arb = rot_z(37) @ rot_x(-110) @ rot_y(20)
    pts_c, q_c, t_c = to_colmap_frame(pts, qs, ts, 0.23, R_arb, np.array([4.0, -1.0, 9.0]))
    imgs = Images([f"{i}.jpg" for i in range(len(q_c))], q_c, t_c)
    points = Points(pts_c, np.full(len(pts_c), 0.5), np.full(len(pts_c), 5))

    al = compute_alignment(imgs, points, camera_height_m=1.5, floor_percentile=2.0)
    assert al.up_consistency > 0.95
    out = al.apply(pts_c)
    # Sol à 0, hauteur des murs ~2,5 m (échelle retrouvée via la hauteur caméra).
    assert abs(np.percentile(out[:, 1], 2)) < 0.05
    assert np.percentile(out[:, 1], 99.5) == pytest.approx(2.5, abs=0.15)
    # Murs alignés sur les axes, plus grande dimension le long de Z.
    ext = np.percentile(out, 99, axis=0) - np.percentile(out, 1, axis=0)
    assert ext[2] == pytest.approx(6.0, abs=0.3)
    assert ext[0] == pytest.approx(4.0, abs=0.3)
    # Centré.
    assert abs(np.median(out[:, 0])) < 0.2 and abs(np.median(out[:, 2])) < 0.3
    # Diagnostics : caméras tenues droites (roulis ~0), échelle = 1 / 0.23.
    d = al.diagnostics()
    assert d["camera_height_assumed_m"] == 1.5
    assert d["camera_roll_median_deg"] < 1
    assert d["rotated_images"]["0"] == len(q_c)
    assert d["scale_m_per_unit"] == pytest.approx(1 / 0.23, rel=0.05)
    assert 0 <= d["vertical_angle_deg"] <= 180


def test_verticale_robuste_portrait_et_tangage():
    """Mélange paysage/portrait (comme playroom) + caméras qui filment le sol : la verticale
    doit rester exacte (la moyenne des « haut » se trompe dans ce cas)."""
    from pipeline.alignment import estimate_vertical
    rng = np.random.default_rng(3)
    n, qs = 200, []
    flip = np.diag([1.0, -1.0, -1.0])
    for i in range(n):
        pitch = -50 if i % 3 == 0 else rng.normal(0, 10)          # 1/3 regarde fortement vers le bas
        roll_img = 90 if i % 4 == 0 else 0                        # 1/4 en portrait
        cam_to_world = rot_y(rng.uniform(0, 360)) @ rot_x(pitch) @ flip @ rot_z(roll_img)
        qs.append(_rot_to_quat(cam_to_world.T))
    R_arb = rot_z(25) @ rot_x(70)
    imgs = Images([str(i) for i in range(n)],
                  np.array([_rot_to_quat(quat_to_rot_one(q) @ R_arb.T) for q in qs]), np.zeros((n, 3)))
    v = estimate_vertical(imgs)
    true_up = R_arb @ np.array([0, 1.0, 0])
    err = np.degrees(np.arccos(np.clip(v["up"] @ true_up, -1, 1)))
    assert err < 0.5, err
    assert v["orientations"]["0"] + v["orientations"]["180"] == 150
    assert np.median(v["roll"]) < 0.5


def quat_to_rot_one(q):
    from pipeline.colmap_text import quat_to_rot
    return quat_to_rot(np.array([q]))[0]


def _identity():
    return Alignment(scale=1.0, R=np.eye(3).tolist(), t=[0.0, 0.0, 0.0], up_consistency=1,
                     camera_height_model=1, yaw_deg=10, floor_model=0)


def test_overrides_manuels():
    from pipeline.alignment import apply_overrides
    al = apply_overrides(_identity(), scale_factor=2, yaw_deg=90, floor_offset_m=0.1)
    out = al.apply(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(out, [[0, 0.1, -2], [0, 0.1, 0]], atol=1e-9)
    assert al.yaw_deg == 100 and al.overrides == {"scale_factor": 2.0, "yaw_deg": 90.0, "floor_offset_m": 0.1}
    # L'échelle seule garde le sol à 0.
    al2 = apply_overrides(_identity(), scale_factor=1.3)
    assert np.allclose(al2.apply(np.array([[2.0, 0.0, -1.0]])), [[2.6, 0, -1.3]])


def test_overrides_lus_depuis_la_config(tmp_path):
    import json
    from pipeline.config import load_config
    from pipeline.context import Context
    from pipeline.steps.s5_export import load_alignment

    p = tmp_path / "c.yaml"
    p.write_text("alignment:\n  override:\n    yaw_deg: 45\n")
    ctx = Context(slug="s", cfg=load_config(p), root=tmp_path / "s")
    (ctx.root / "3_sfm").mkdir(parents=True)
    (ctx.root / "3_sfm" / "alignment.json").write_text(json.dumps(_identity().to_dict()))
    al = load_alignment(ctx)
    assert al.yaw_deg == 55 and al.overrides["yaw_deg"] == 45.0


@pytest.mark.parametrize("angles", [(10, 20, 30), (-80, 45, 170), (0, 89.9, 0), (120, -30, -60)])
def test_euler_zyx(angles):
    a, b, g = angles
    R = rot_z(g) @ rot_y(b) @ rot_x(a)
    a2, b2, g2 = euler_zyx(R)
    assert np.allclose(rot_z(g2) @ rot_y(b2) @ rot_x(a2), R, atol=1e-6)


def test_actions_splat_transform_reelles(tmp_path, splat_transform):
    """La traduction -r/-s/-t doit donner exactement s·R·X + t (mesuré sur splat-transform)."""
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-3, 3, (200, 3))
    R = rot_z(-33) @ rot_y(71) @ rot_x(152)
    al = Alignment(scale=0.37, R=R.tolist(), t=[1.5, -2.0, 0.7], up_consistency=1, camera_height_model=1,
                   yaw_deg=0, floor_model=0)
    src, dst = tmp_path / "in.ply", tmp_path / "out.ply"
    write_gaussian_ply(src, xyz)
    run_st(splat_transform, "-q", "-w", src, *splat_transform_args(al), dst)
    c = read_vertex_columns(dst, ["x", "y", "z"])
    got = np.stack([c["x"], c["y"], c["z"]], 1)
    assert np.allclose(got, al.apply(xyz), atol=1e-3)


def test_controle_du_sol_mesure_l_inclinaison():
    """Le contrôle par le nuage de points est indépendant des caméras : un sol penché de 4°
    doit être mesuré à ~4° (« à surveiller »), un sol plat à ~0° (« ok »)."""
    from pipeline.alignment import floor_plane_check
    pts, *_ = synthetic_scene()
    pts = pts + np.random.default_rng(5).normal(0, 0.005, pts.shape)
    flat = floor_plane_check(pts)
    assert flat["tilt_deg"] < 0.5 and flat["status"] == "ok"
    assert abs(flat["plane_height_m"]) < 0.02
    tilted = floor_plane_check(pts @ rot_x(4).T)
    assert tilted["tilt_deg"] == pytest.approx(4, abs=0.5) and tilted["status"] == "à surveiller"
    assert floor_plane_check(pts @ rot_z(7).T)["status"] == "AVERTISSEMENT"


def test_alignement_contient_le_controle_du_sol():
    pts, centers, qs, ts = synthetic_scene()
    pts_c, q_c, t_c = to_colmap_frame(pts, qs, ts, 0.5, rot_x(-100), np.array([1.0, 2.0, 3.0]))
    al = compute_alignment(Images([str(i) for i in range(len(q_c))], q_c, t_c),
                           Points(pts_c, np.full(len(pts_c), 0.5), np.full(len(pts_c), 5)))
    fc = al.diagnostics()["floor_check"]
    assert fc["tilt_deg"] < 1 and fc["status"] == "ok"


def test_modele_reecrit_dans_le_repere_d_entrainement(tmp_path):
    """transform_model_txt : poses et points transformés de façon cohérente (reprojection
    identique), lignes POINTS2D et caméras inchangées."""
    from pipeline.colmap_text import camera_centers_and_up, read_images, read_points, transform_model_txt
    pts, centers, qs, ts = synthetic_scene()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
    lines = ["# Image list"]
    for i, (q, t) in enumerate(zip(qs, ts)):
        lines += [f"{i + 1} {' '.join(f'{v:.17g}' for v in q)} {' '.join(f'{v:.17g}' for v in t)} 1 img{i}.jpg",
                  "" if i % 2 else "10.5 20.5 -1"]
    (src / "images.txt").write_text("\n".join(lines) + "\n")
    (src / "points3D.txt").write_text("\n".join(f"{j + 1} {x} {y} {z} 1 2 3 0.5 1 0 2 0"
                                                for j, (x, y, z) in enumerate(pts[:50])) + "\n")
    s, R, t = 0.37, rot_z(20) @ rot_x(-70), np.array([1.0, -2.0, 0.5])
    transform_model_txt(src, dst, s, R, t)
    a, b = read_images(src / "images.txt"), read_images(dst / "images.txt")
    c_old, _ = camera_centers_and_up(a)
    c_new, _ = camera_centers_and_up(b)
    assert np.allclose(c_new, s * c_old @ R.T + t, atol=1e-6)
    P_old, P_new = read_points(src / "points3D.txt").xyz, read_points(dst / "points3D.txt").xyz
    assert np.allclose(P_new, s * P_old @ R.T + t, atol=1e-5)
    from pipeline.colmap_text import quat_to_rot
    Ro, Rn = quat_to_rot(a.qvec), quat_to_rot(b.qvec)
    for i in range(0, len(a.names), 17):   # même point vu dans la caméra : même direction
        x_old = Ro[i] @ P_old[0] + a.tvec[i]
        x_new = Rn[i] @ P_new[0] + b.tvec[i]
        assert np.allclose(x_new, s * x_old, atol=1e-5)
    txt = (dst / "images.txt").read_text().splitlines()
    assert txt[2] == "10.5 20.5 -1" and txt[4] == ""
    assert (dst / "cameras.txt").read_text() == (src / "cameras.txt").read_text()


def test_images_inexploitables_ecartees(tmp_path):
    """Images sans point 3D ou à une distance absurde : écartées du compte et du modèle d'entraînement,
    y compris dans les TRACK des points 3D."""
    from pipeline.colmap_text import read_images, read_points, transform_model_txt, usable_mask
    pts, centers, qs, ts = synthetic_scene()
    n = 30
    ts = ts[:n].copy()
    ts[5] = ts[5] + np.array([0, 0, 5e5])       # pose absurde
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
    lines = []
    for i in range(n):
        obs = "" if i == 7 else " ".join(f"{x}.5 {x}.5 {x + 1}" for x in range(20))  # image 8 : aucun point 3D
        lines += [f"{i + 1} {' '.join(f'{v:.17g}' for v in qs[i])} {' '.join(f'{v:.17g}' for v in ts[i])} 1 i{i}.jpg",
                  obs]
    (src / "images.txt").write_text("\n".join(lines) + "\n")
    (src / "points3D.txt").write_text("1 0 0 0 1 2 3 0.5 6 0 8 0 9 3\n2 1 1 1 1 2 3 0.5 8 1\n")
    imgs = read_images(src / "images.txt")
    assert imgs.n_points3d[7] == 0 and imgs.n_points3d[0] == 20
    ok = usable_mask(imgs, min_points3d=15, max_distance_ratio=10)
    assert not ok[5] and not ok[7] and ok.sum() == n - 2
    transform_model_txt(src, dst, 1.0, np.eye(3), np.zeros(3), keep=set(imgs.subset(ok).names))
    out = read_images(dst / "images.txt")
    assert len(out.names) == n - 2 and "i5.jpg" not in out.names and "i7.jpg" not in out.names
    pts3d = (dst / "points3D.txt").read_text().splitlines()
    assert pts3d == ["1 0 0 0 1 2 3 0.5 9 3"]   # entrées des images 6 et 8 retirées ; point 2 supprimé
    assert len(read_points(dst / "points3D.txt").xyz) == 1


def _read_bin_independant(d):
    """Lecteur binaire COLMAP minimal, écrit d'après la doc (indépendant de write_bin_for_gsplat)."""
    import struct
    nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
    cams, imgs, pts = {}, {}, {}
    with open(d / "cameras.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            cams[cid] = (model, w, h, struct.unpack(f"<{nparams[model]}d", f.read(8 * nparams[model])))
    with open(d / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid, *qt, cid = struct.unpack("<I7dI", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            obs = [struct.unpack("<ddq", f.read(24)) for _ in range(n2d)]
            imgs[name.decode()] = (iid, np.array(qt[:4]), np.array(qt[4:]), cid, obs)
    with open(d / "points3D.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            pid, x, y, z, r, g, b, err = struct.unpack("<Q3d3Bd", f.read(43))
            tl = struct.unpack("<Q", f.read(8))[0]
            pts[pid] = (np.array([x, y, z]), (r, g, b), err, struct.unpack(f"<{2 * tl}i", f.read(8 * tl)))
        assert f.read() == b""
    return cams, imgs, pts


def test_aller_retour_binaire(tmp_path):
    """write_bin_for_gsplat relu par un lecteur indépendant : intrinsèques, poses, observations et
    points identiques au TXT source."""
    from pipeline.colmap_text import read_images, write_bin_for_gsplat
    pts, centers, qs, ts = synthetic_scene()
    src = tmp_path / "txt"
    src.mkdir()
    (src / "cameras.txt").write_text("1 PINHOLE 1264 832 1040.25 1041.5 632.1 415.9\n"
                                     "2 SIMPLE_RADIAL 800 600 700 400 300 -0.01\n")
    lines = []
    for i in range(12):
        lines += [f"{i + 1} {' '.join(f'{v:.17g}' for v in qs[i])} {' '.join(f'{v:.17g}' for v in ts[i])} "
                  f"{1 + i % 2} img {i}.jpg", "" if i == 3 else f"{i}.25 {i}.75 {i + 1} 5.5 6.5 -1"]
    (src / "images.txt").write_text("\n".join(lines) + "\n")
    (src / "points3D.txt").write_text("\n".join(f"{j + 1} {x:.17g} {y:.17g} {z:.17g} 10 20 30 0.75 {j + 1} 0"
                                                for j, (x, y, z) in enumerate(pts[:12])) + "\n")
    write_bin_for_gsplat(src, tmp_path / "bin")
    cams, imgs, p3d = _read_bin_independant(tmp_path / "bin")
    assert cams[1] == (1, 1264, 832, (1040.25, 1041.5, 632.1, 415.9))
    assert cams[2] == (2, 800, 600, (700.0, 400.0, 300.0, -0.01))
    ref = read_images(src / "images.txt")
    for k, name in enumerate(ref.names):
        iid, q, t, cid, obs = imgs[name]
        assert iid == k + 1 and cid == 1 + k % 2 and name == f"img {k}.jpg"
        assert np.allclose(q, ref.qvec[k], atol=1e-15) and np.allclose(t, ref.tvec[k], atol=1e-15)
        assert obs == ([] if k == 3 else [(k + 0.25, k + 0.75, k + 1), (5.5, 6.5, -1)])
    for j in range(12):
        xyz, rgb, err, track = p3d[j + 1]
        assert np.allclose(xyz, pts[j], atol=1e-12) and rgb == (10, 20, 30) and err == 0.75
        assert track == (j + 1, 0)
