import base64
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import quality
from pipeline.config import ConfigError, load_config
from pipeline.render_check import (load_splat_arrays, mean_metrics, pick_train_views, psnr, sheet_cameras,
                                   split_indices)
from pipeline.steps import s8_notify


def _write_ply(path: Path, cols: dict[str, np.ndarray]) -> None:
    names = list(cols)
    n = len(cols[names[0]])
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(
        f"property float {c}\n" for c in names) + "end_header\n"
    data = np.stack([cols[c] for c in names], 1).astype("<f4")
    path.write_bytes(hdr.encode() + data.tobytes())


def test_split_identique_a_gsplat():
    tr, te = split_indices(20, 8)
    assert te.tolist() == [0, 8, 16]
    assert len(tr) == 17 and 0 not in tr
    tr, te = split_indices(5, 0)
    assert len(te) == 0 and len(tr) == 5


def test_vues_entrainement_reparties():
    tr, _ = split_indices(185, 8)
    v = pick_train_views(tr, 3)
    assert len(v) == 3 and all(i % 8 != 0 for i in v)
    assert v[0] < 185 / 3 < v[1] < 2 * 185 / 3 < v[2]
    assert pick_train_views(tr[:2], 3) == sorted(set(tr[:2].tolist()))
    assert pick_train_views(tr[:0], 3) == []


def test_lecture_ply_sh_et_echelle(tmp_path):
    # 1 gaussienne, SH degré 1 : f_rest canal par canal (R0 R1 R2 G0 G1 G2 B0 B1 B2)
    cols = {"x": [1.0], "y": [2.0], "z": [3.0], "rot_0": [2.0], "rot_1": [0.0], "rot_2": [0.0], "rot_3": [0.0],
            "scale_0": [np.log(0.1)], "scale_1": [np.log(0.2)], "scale_2": [np.log(0.3)], "opacity": [0.0],
            "f_dc_0": [0.1], "f_dc_1": [0.2], "f_dc_2": [0.3]}
    for i in range(9):
        cols[f"f_rest_{i}"] = [float(i)]
    _write_ply(tmp_path / "a.ply", {k: np.asarray(v, float) for k, v in cols.items()})
    s = load_splat_arrays(tmp_path / "a.ply", scale=2.0)
    assert s["sh_degree"] == 1
    assert np.allclose(s["means"], [[2, 4, 6]])
    assert np.allclose(s["scales"], [[0.2, 0.4, 0.6]])          # exp(log) x 2  <=>  log + ln 2
    assert np.allclose(s["quats"], [[1, 0, 0, 0]])
    assert np.allclose(s["opacities"], [0.5])
    assert np.allclose(s["sh"][0, 0], [0.1, 0.2, 0.3])
    # coefficient k de la bande 1, canal c = f_rest[c * 3 + k]  (= shN.permute(0, 2, 1) de gsplat)
    assert np.allclose(s["sh"][0, 1:], [[0, 3, 6], [1, 4, 7], [2, 5, 8]])


def test_psnr_et_moyennes():
    a = np.zeros((4, 4, 3))
    assert psnr(a, a) == 99.0
    assert psnr(a, a + 0.1) == pytest.approx(20.0)
    views = {"a": {"psnr": 20.0, "ssim": 0.8}, "b": {"psnr": 30.0, "ssim": None}}
    assert mean_metrics(views, ["a", "b"]) == {"psnr": 25.0, "ssim": 0.8, "n": 2}


def test_cameras_planche():
    cams = sheet_cameras([1.0, 1.5, 2.0], 0.5, 70, 640, 480)
    assert len(cams) == 4
    for w2c, K in cams:
        c = -w2c[:3, :3].T @ w2c[:3, 3]                     # centre = eye x échelle
        assert np.allclose(c, [0.5, 0.75, 1.0])
        assert np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3))
        assert w2c[1, 1] < 0                                # y caméra (bas) opposé au y monde (haut)
    assert K[0, 2] == 320 and K[1, 2] == 240


def test_seuil_baisse_psnr():
    tr = {"test": {"psnr": 28.0}}
    assert quality.export_drop_reason(tr, {"test": {"psnr": 27.2}}, "normal", 1.5) == (0.8, None)
    drop, reason = quality.export_drop_reason(tr, {"test": {"psnr": 25.0}}, "normal", 1.5)
    assert drop == 3.0 and "normal" in reason and "1.5" in reason
    # sans vue de test : repli sur les vues d'entraînement
    tr2 = {"test": {"n": 0}, "train_views": {"psnr": 30.0}}
    assert quality.export_drop_reason(tr2, {"test": {}, "train_views": {"psnr": 29.0}}, "mobile", 4) == (1.0, None)
    assert quality.export_drop_reason({}, {}, "normal", 1) == (None, None)


def test_config_seuils_par_variante(tmp_path):
    cfg = load_config(None)
    assert cfg["train"]["test_every"] == 8
    assert set(cfg["quality"]["max_psnr_drop_db"]) == set(cfg["export"]["variants"])
    p = tmp_path / "c.yaml"
    p.write_text("quality:\n  max_psnr_drop_db:\n    normal: 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_psnr_drop_db.mobile"):
        load_config(p)


def test_email_qualite_et_piece_jointe(tmp_path):
    sheet = tmp_path / "normal_planche.png"
    sheet.write_bytes(b"\x89PNG fake")
    m = {"psnr": 27.5, "ssim": 0.9, "lpips": 0.12,
         "quality_train": {"test": {"psnr": 27.4, "ssim": 0.9, "lpips": 0.12, "n": 24}, "train_views": {"psnr": 31.0}},
         "quality_normal": {"test": {"psnr": 27.0, "n": 24}, "psnr_drop_db": 0.4},
         "quality_mobile": {"error": "boom"}, "quality_sheet": str(sheet)}

    class Ctx:
        cfg = load_config(None)

    text = "\n".join(s8_notify.quality_lines(Ctx(), m))
    assert "PSNR 27.50 dB, SSIM 0.900, LPIPS 0.120" in text
    assert "baisse +0.40 dB (max 1.5)" in text
    assert "mobile : impossible (boom)" in text
    assert "normal_planche.png" in text
    att = s8_notify.attachments(m)
    assert att == [{"filename": "normal_planche.png", "content": base64.b64encode(b"\x89PNG fake").decode()}]
    assert s8_notify.attachments({}) == []


def test_controle_en_echec_non_bloquant(tmp_path, monkeypatch):
    from pipeline.context import StepError

    class Ctx:
        dry_run = False

        def run(self, cmd, cwd=None):
            raise StepError("CUDA indisponible")

    res = quality.run_check(Ctx(), ["x"], tmp_path / "check", "train")
    assert res == {"error": "CUDA indisponible"}
    assert quality.summary(res) == {"error": "CUDA indisponible"}


def test_resume_json(tmp_path):
    res = {"test": {"psnr": 1}, "train_views": {}, "gaussians": 3, "sh_degree": 2, "sheet": "x.png", "views": {}}
    assert quality.summary(res) == {"test": {"psnr": 1}, "train_views": {}, "gaussians": 3, "sh_degree": 2,
                                    "sheet": "x.png"}
    json.dumps(quality.summary(res))


def test_commande_coordonnees_negatives(tmp_path):
    """Régression : « --sheet-eye -0.8,... » était pris pour une option par argparse."""
    from pipeline.render_check import build_parser

    class Ctx:
        cfg = load_config(None)

    cmd = quality.command(Ctx(), tmp_path / "a.ply", "normal", tmp_path, tmp_path, scale=0.85,
                          reference=tmp_path / "ref", sheet_eye=[-0.8, 1.6, -0.5], sheet_scale=0.85)
    a = build_parser().parse_args(cmd[3:])
    assert a.sheet_eye == "-0.8000,1.6000,-0.5000" and a.scale == 0.85 and a.test_every == 8
