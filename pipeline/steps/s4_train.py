"""Étape 4 — entraînement Gaussian Splatting avec gsplat (stratégie MCMC), sortie .ply.

Utilise examples/simple_trainer.py de gsplat v1.5.3 (cloné dans l'image Docker) :
  python simple_trainer.py mcmc --data_dir <undistorted> ...
Points vérifiés dans le code source v1.5.3 :
  - data_factor vaut 4 par défaut (cherche images_4/) : on force 1 ;
  - normalize_world_space vaut True par défaut et le .ply est alors exporté dans l'espace
    normalisé : on le désactive pour rester dans le repère COLMAP (aligné à l'étape 5) ;
  - le .ply final s'appelle ply/point_cloud_<max_steps-1>.ply ;
  - le parser lit sparse/0 puis, à défaut, sparse/ (sortie de image_undistorter).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from ..context import Context, StepError
from .. import quality
from ..train_guard import TrainGuard
from ..plyutil import read_header
from . import s3_colmap

NAME = "train"
log = logging.getLogger("pipeline")


def train_dir(ctx: Context) -> Path:
    return ctx.root / "4_train"


def splats_path(ctx: Context) -> Path:
    return train_dir(ctx) / "splats.ply"


def check_dir(ctx: Context) -> Path:
    return train_dir(ctx) / "check"


def check_renders(ctx: Context) -> Path:
    """Rendus bruts du .ply entraîné : référence du contrôle après export (étape 5)."""
    return check_dir(ctx) / "renders" / quality.TRAIN_TAG


def outputs(ctx: Context) -> list[Path]:
    return [splats_path(ctx)]


def build_command(cfg: dict, data_dir: Path, result_dir: Path) -> list[str]:
    t = cfg["train"]
    last = str(t["max_steps"])
    g = t["guard"]
    # .ply partiels aux points de contrôle du garde-fou : rapatriés si l'entraînement est arrêté.
    ply_steps = sorted({*(int(c) for c in g["checkpoints"] if g["enabled"] and int(c) < t["max_steps"]),
                        t["max_steps"]})
    return [
        t["python"], t["trainer"], "mcmc",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--data_factor", str(t["data_factor"]),
        "--max_steps", last,
        "--strategy.cap-max", str(t["cap_max"]),
        "--test_every", str(t["test_every"]),
        "--eval_steps", last,
        "--save_steps", last,
        "--ply_steps", *[str(x) for x in ply_steps],
        "--save_ply",
        "--disable_viewer",
        "--disable_video",
        "--no-normalize-world-space",
        *[str(a) for a in t["extra_args"]],
    ]


def _flag_names(cmd: list[str]) -> list[str]:
    return [a for a in cmd[3:] if a.startswith("--")]


def check_flags(help_text: str, cmd: list[str]) -> list[str]:
    """Options de la commande absentes de l'aide du trainer (tyro accepte _ et -)."""
    norm_help = help_text.replace("_", "-")
    missing = []
    for flag in _flag_names(cmd):
        f = flag.replace("_", "-")
        if not re.search(re.escape(f) + r"(?![\w-])", norm_help):
            missing.append(flag)
    return missing


def run(ctx: Context) -> dict:
    data_dir = s3_colmap.train_data_dir(ctx)
    d = train_dir(ctx)
    result_dir = d / "results"
    if result_dir.exists() and not ctx.dry_run:
        shutil.rmtree(result_dir)
    ctx.dir("4_train")
    cmd = build_command(ctx.cfg, data_dir, result_dir)

    # Vérifie les options avant de consommer des heures de GPU.
    if not ctx.dry_run:
        help_text = ctx.capture([cmd[0], cmd[1], "mcmc", "--help"])
        missing = check_flags(help_text, cmd)
        if missing:
            raise StepError(f"options inconnues du trainer gsplat : {missing}\n{help_text[-1500:]}")

    g = ctx.cfg["train"]["guard"]
    guard = None
    if g["enabled"]:
        guard = TrainGuard(checkpoints=list(g["checkpoints"]), loss_rise_ratio=g["loss_rise_ratio"],
                           loss_rise_min=g["loss_rise_min"], max_dead_fraction=g["max_dead_fraction"],
                           min_gaussians_ratio=g["min_gaussians_ratio"],
                           scene_scale_range=tuple(g["scene_scale_range"]))
    try:
        ctx.run(cmd, cwd=Path(ctx.cfg["train"]["trainer"]).parent, on_line=guard.feed if guard else None)
    finally:
        if guard and not ctx.dry_run:
            (train_dir(ctx) / "guard.json").write_text(json.dumps(guard.report, indent=2), encoding="utf-8")
            log.info("Garde-fou : %s", guard.report)
    if ctx.dry_run:
        if quality.enabled(ctx):
            quality.run_check(ctx, quality.command(ctx, splats_path(ctx), quality.TRAIN_TAG, check_dir(ctx),
                                                   data_dir), check_dir(ctx), quality.TRAIN_TAG)
        return {}

    plys = sorted((result_dir / "ply").glob("point_cloud_*.ply"),
                  key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not plys:
        raise StepError(f"aucun .ply produit dans {result_dir / 'ply'}")
    last_step = int(re.findall(r"\d+", plys[-1].stem)[-1])
    if last_step != ctx.cfg["train"]["max_steps"] - 1:
        raise StepError(f"dernier .ply à l'itération {last_step}, attendu {ctx.cfg['train']['max_steps'] - 1} "
                        "(seuls des .ply partiels existent)")
    shutil.copyfile(plys[-1], splats_path(ctx))

    metrics: dict = {"gaussians_trained": read_header(splats_path(ctx)).count()}
    if guard:
        metrics["guard"] = guard.report
    stats = sorted((result_dir / "stats").glob("val_step*.json"))
    if stats:
        try:
            val = json.loads(stats[-1].read_text())
            for k in ("psnr", "ssim", "lpips"):
                if k in val:
                    metrics[k] = round(float(val[k]), 4 if k != "psnr" else 2)
        except (KeyError, ValueError):
            pass
    if quality.enabled(ctx):
        cd = check_dir(ctx)
        if cd.exists():
            shutil.rmtree(cd)
        res = quality.run_check(ctx, quality.command(ctx, splats_path(ctx), quality.TRAIN_TAG, cd, data_dir),
                                cd, quality.TRAIN_TAG)
        metrics["quality_train"] = quality.summary(res)
        if "error" in res:
            return metrics
        mine = (res.get("test") or {}).get("psnr")
        log.info("Contrôle visuel : test %s ; vues d'entraînement %s", res.get("test"), res.get("train_views"))
        if mine is not None and "psnr" in metrics and abs(mine - metrics["psnr"]) > 0.5:
            log.warning("PSNR de test du contrôle (%.2f dB) différent de celui du trainer (%.2f dB) : "
                        "lecture du .ply ou caméras à vérifier", mine, metrics["psnr"])
    return metrics


def verify(ctx: Context, metrics: dict) -> None:
    p = splats_path(ctx)
    if not p.is_file():
        raise StepError("splats.ply absent")
    h = read_header(p)
    n = h.count()
    props = h.props()
    for col in ("x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"):
        if col not in props:
            raise StepError(f"colonne {col} absente de splats.ply")
    if n < 10_000:
        raise StepError(f"seulement {n} gaussiennes : entraînement raté")
    cap = ctx.cfg["train"]["cap_max"]
    if n > cap * 1.01:
        raise StepError(f"{n} gaussiennes > cap_max {cap}")
