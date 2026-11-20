"""Tests sans GPU : parsing COLMAP, options gsplat, tour-entry, rapport, reprise, dry-run."""

from pathlib import Path

import numpy as np
import pytest

from pipeline import cli
from pipeline.colmap_text import read_images, read_points
from pipeline.config import load_config
from pipeline.context import Context, StepError
from pipeline.runner import check_resumable, run_pipeline
from pipeline.state import State
from pipeline.steps import s4_train, s7_tour_entry, s8_notify

IMAGES_TXT = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3, mean observations per image: 1
1 0.9 0.1 0.2 0.3 1 2 3 1 frame_0001.jpg
10.0 20.0 5 11.0 21.0 -1
2 1 0 0 0 0.5 0.5 0.5 1 frame 0002.jpg

3 1 0 0 0 0 0 0 1 frame_0003.jpg

"""

POINTS_TXT = """# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
5 1.0 2.0 3.0 255 0 0 0.7 1 0 2 4 3 1
6 -1 -2 -3 0 0 0 1.5 1 1
"""


def test_lecture_modele_colmap_texte(tmp_path):
    (tmp_path / "images.txt").write_text(IMAGES_TXT)
    (tmp_path / "points3D.txt").write_text(POINTS_TXT)
    imgs = read_images(tmp_path / "images.txt")
    assert imgs.names == ["frame_0001.jpg", "frame 0002.jpg"]  # l'image 3 (pose identité) est ignorée
    assert imgs.tvec[1].tolist() == [0.5, 0.5, 0.5]
    pts = read_points(tmp_path / "points3D.txt")
    assert pts.xyz.shape == (2, 3) and pts.track_len.tolist() == [3, 1]


def test_options_gsplat_verifiees():
    cfg = load_config(None)
    cmd = s4_train.build_command(cfg, Path("data"), Path("res"))
    help_ok = " ".join(f.replace("_", "-") for f in s4_train._flag_names(cmd))
    assert s4_train.check_flags(help_ok, cmd) == []
    assert "--strategy.cap-max" in s4_train.check_flags(help_ok.replace("--strategy.cap-max", ""), cmd)
    assert cmd[cmd.index("--data_factor") + 1] == "1"
    assert "--no-normalize-world-space" in cmd


def test_tour_entry():
    cfg = load_config(None)
    rng = np.random.default_rng(0)
    xyz = np.c_[rng.uniform(-2, 2, 50000), rng.uniform(0, 2.5, 50000), rng.uniform(-3, 4, 50000)]
    xyz[:100] *= 500  # flotteurs lointains : ignorés par les centiles
    layout = s7_tour_entry.compute_layout(xyz, cfg)
    b = layout["bounds"]
    assert b["min"][1] == 1.0 and b["max"][1] == 2.2
    assert -1.8 < b["min"][0] < -1.5 and 1.5 < b["max"][0] < 1.8
    text = s7_tour_entry.render_entry("studio-test", "tours/studio-test.spz", "tours/studio-test-mobile.spz",
                                      layout)
    assert 'slug: "studio-test",' in text
    assert 'sceneMobile: "tours/studio-test-mobile.spz",' in text
    assert 'label: "Vue d\'ensemble"' in text
    assert "bounds: { min: [" in text


def _ctx(tmp_path, **kw):
    return Context(slug="studio-test", cfg=load_config(None), root=tmp_path / "studio-test", **kw)


def test_rapport_email(tmp_path):
    ctx = _ctx(tmp_path)
    st = State(tmp_path / "s.json", persist=False)
    st.start(3, "colmap")
    st.finish(3, 754, {"aligned_images": 380, "frames_total": 400, "aligned_ratio": 0.95, "mapper": "global"})
    st.start(5, "export")
    st.finish(5, 60, {"normal_splats": 998000, "normal_sh_bands": 1, "normal_size_mb": 14.2})
    subject, text = s8_notify.build_report(ctx, st)
    assert "studio-test" in subject
    assert "380/400" in text and "12 min 34 s" in text
    assert "https://visite.stayphora.fr/studio-test" in text


def test_reprise_refusee_si_etape_precedente_absente(tmp_path):
    ctx = _ctx(tmp_path)
    st = State(tmp_path / "s.json", persist=False)
    with pytest.raises(StepError, match="1. fetch"):
        check_resumable(ctx, st, 3)
    st.start(1, "fetch"); st.finish(1, 1, {})
    st.start(2, "frames"); st.finish(2, 1, {})
    with pytest.raises(StepError, match="absents"):
        check_resumable(ctx, st, 3)  # state OK mais fichiers manquants


def test_etapes_locales_et_reprise(tmp_path):
    """Étapes 1 puis 7 en vrai (sans GPU) : copie de la vidéo, tour-entry depuis un cleaned.ply."""
    from tests.conftest import write_gaussian_ply

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\0" * 2_000_000)
    ctx = _ctx(tmp_path, input=str(video))
    st = run_pipeline(ctx, from_step=1, to_step=1)
    assert st.is_done(1) and (ctx.root / "1_input" / "video.mp4").is_file()
    assert any(ctx.logs_dir.glob("*_1_fetch.log"))

    # Simule les étapes 2 à 6 terminées puis relance seulement l'étape 7.
    data = State.load(ctx.root / "state.json")
    for n in range(2, 7):
        data.start(n, "x"); data.finish(n, 1, {})
    (ctx.root / "2_frames" / "images").mkdir(parents=True)
    und = ctx.root / "3_sfm" / "undistorted"
    (und / "images").mkdir(parents=True); (und / "sparse").mkdir()
    (ctx.root / "3_sfm" / "alignment.json").write_text("{}")
    tr = ctx.root / "3_sfm" / "train"
    (tr / "images").mkdir(parents=True); (tr / "sparse" / "0").mkdir(parents=True)
    (tr / "sparse" / "0" / "images.bin").write_text(""); (tr / "transform.json").write_text("{}")
    (ctx.root / "4_train").mkdir(); (ctx.root / "4_train" / "splats.ply").write_text("")
    exp = ctx.root / "5_export"; exp.mkdir()
    rng = np.random.default_rng(0)
    write_gaussian_ply(exp / "cleaned.ply", np.c_[rng.uniform(-2, 2, 5000), rng.uniform(0, 2.5, 5000),
                                                  rng.uniform(-3, 3, 5000)])
    (exp / "studio-test.spz").write_text(""); (exp / "studio-test-mobile.spz").write_text("")

    ctx2 = _ctx(tmp_path)
    st2 = run_pipeline(ctx2, from_step=7, to_step=7)
    assert st2.is_done(7) and st2.is_done(6)
    assert 'slug: "studio-test"' in (ctx.root / "tour-entry.ts").read_text(encoding="utf-8")


def test_dry_run_complet_ne_cree_rien(tmp_path, capsys):
    code = cli.main(["run", "--input", "https://example.com/v.mp4", "--slug", "studio-test",
                     "--work-dir", str(tmp_path / "w"), "--dry-run"])
    assert code == 0
    assert not (tmp_path / "w").exists()
    out = capsys.readouterr().out
    for needle in ("sharp-frames", "global_mapper", "view_graph_calibrator", "simple_trainer.py mcmc",
                   "--decimate 500000", "r2://stayphora-assets/tours/studio-test.spz", "contact@stayphora.fr"):
        assert needle in out, needle


def test_cli_slug_invalide(tmp_path):
    assert cli.main(["run", "--input", "x.mp4", "--slug", "Mon_Studio", "--dry-run"]) == 2


def test_secrets_masques(monkeypatch):
    from pipeline.context import fmt_cmd
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "super-secret-value-123")
    assert "super-secret" not in fmt_cmd(["aws", "--key", "super-secret-value-123"])


def _replay(guard, path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stop = guard.feed(line)
        if stop:
            return stop, guard
    return None, guard


def test_garde_fou_arrete_l_essai_diverge_du_20_11():
    """Journal réel de l'essai RunPod raté : arrêt immédiat sur l'échelle de scène (840)."""
    from pipeline.train_guard import TrainGuard
    stop, g = _replay(TrainGuard(), Path(__file__).parent / "data" / "train_divergence.log")
    assert stop and "échelle de scène" in stop and g.step < 10


def test_garde_fou_perte_et_gaussiennes_mortes_a_1000():
    """Même journal, sans le contrôle d'échelle : arrêt à 1 000 sur la perte et les relocalisations."""
    from pipeline.train_guard import TrainGuard
    stop, g = _replay(TrainGuard(scene_scale_range=(0, 1e9)), Path(__file__).parent / "data" / "train_divergence.log")
    assert stop and "itération 1000" in stop, stop
    assert "71 % des gaussiennes mortes" in stop, stop
    assert 1000 <= g.step < 1100
    assert g.report["step_1000"]["relocated_fraction"] > 0.5


def test_garde_fou_perte_seule_a_3000():
    """Sans le critère des gaussiennes mortes, la remontée de la perte arrête l'essai à 3 000
    (à 1 000 elle n'est encore qu'à 1,26 x son minimum : pas d'arrêt)."""
    from pipeline.train_guard import TrainGuard
    stop, g = _replay(TrainGuard(scene_scale_range=(0, 1e9), max_dead_fraction=1.1),
                      Path(__file__).parent / "data" / "train_divergence.log")
    assert stop and "itération 3000" in stop and "perte remontée" in stop, stop
    r = g.report["step_3000"]
    assert r["loss_median"] > 2 * r["loss_min"]


def test_garde_fou_laisse_passer_un_entrainement_sain():
    from pipeline.train_guard import TrainGuard
    g = TrainGuard()
    rng = np.random.default_rng(0)
    assert g.feed("Scene scale: 2.63") is None
    total = 14000
    for step in range(1, 3200):
        loss = 0.25 * np.exp(-step / 1500) + 0.05 + rng.normal(0, 0.03)
        assert g.feed(f"loss={loss:.3f}| sh degree=1| :  5%|▌ | {step}/30000 [00:10<05:00, 90it/s]") is None
        if step % 100 == 0 and step >= 500:
            total += 700
            assert g.feed(f"Step {step}: Added 700 GSs. Now having {total} GSs.") is None
            assert g.feed(f"Step {step}: Relocated {total // 50} GSs.") is None
    assert set(g.report) == {"scene_scale", "step_1000", "step_3000"}


def test_commande_train_ply_partiels_et_garde_fou_reel(tmp_path):
    """--ply_steps inclut les points de contrôle ; ctx.run arrête le processus si on_line le demande."""
    import sys
    from pipeline.config import load_config
    from pipeline.context import Context, StepError
    from pipeline.steps import s4_train
    cfg = load_config(None)
    cmd = s4_train.build_command(cfg, Path("d"), Path("r"))
    i = cmd.index("--ply_steps")
    assert cmd[i + 1:i + 4] == ["1000", "3000", "30000"]
    ctx = Context(slug="s", cfg=cfg, root=tmp_path / "s")
    ctx.begin_step(4, "train")
    script = "import time\nfor i in range(10**6):\n    print(f'ligne {i}', flush=True)\n    time.sleep(0.001)"
    with pytest.raises(StepError, match="stop à 20"):
        ctx.run([sys.executable, "-c", script], on_line=lambda l: "stop à 20" if l == "ligne 20" else None)
    ctx.end_step()
