"""Étape 5 : filtre d'opacité, décimation, export .spz (avec splat-transform réel si installé)."""

import json
from pathlib import Path

import numpy as np

from pipeline.config import load_config
from pipeline.plyutil import read_header
from pipeline.steps.s5_export import clean_command
from tests.conftest import run_st, write_gaussian_ply


def test_seuil_opacite_passe_en_lineaire():
    """Le seuil de config (opacité réelle) est transmis tel quel, SANS conversion logit :
    splat-transform v3 compare l'opacité linéaire (cf. test réel ci-dessous)."""
    cfg = load_config(None)
    cmd = clean_command(cfg, Path("in.ply"), Path("out.ply"), None)
    i = cmd.index("-V")
    assert cmd[i + 1] == "opacity,gt,0.05"
    assert "--filter-nan" in cmd


def test_filtre_opacite_reel(tmp_path, splat_transform):
    n = 2000
    xyz = np.random.default_rng(0).uniform(-1, 1, (n, 3))
    op = np.where(np.arange(n) < 500, 0.01, 0.5)   # 500 quasi transparentes (1 %)
    xyz[0, 0] = np.nan                             # + 1 gaussienne NaN
    src, dst = tmp_path / "in.ply", tmp_path / "out.ply"
    write_gaussian_ply(src, xyz, op)
    cmd = clean_command(load_config(None), src, dst, None)
    run_st(splat_transform, *cmd[1:])
    assert read_header(dst).count() == 1500


def test_decimation_puis_spz(tmp_path, splat_transform):
    xyz = np.random.default_rng(0).uniform(-1, 1, (3000, 3))
    src = tmp_path / "in.ply"
    write_gaussian_ply(src, xyz)
    run_st(splat_transform, "-q", "-w", src, "--decimate", "1000", tmp_path / "d.ply")
    run_st(splat_transform, "-q", "-w", tmp_path / "d.ply", "--filter-harmonics", "0", tmp_path / "o.spz")
    out = run_st(splat_transform, "-q", tmp_path / "o.spz", "--info", "json", "null")
    info = json.loads(out[out.index("{"):])
    assert info["format"] == "spz" and info["numGaussians"] == 1000 and info["shBands"] == 0
