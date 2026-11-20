"""Contrôle visuel automatique (étapes 4 et 5) : lance pipeline.render_check et interprète le résultat.

Étape 4 : rendu du .ply entraîné (vues de test + 3 vues d'entraînement) -> 4_train/check/train.json.
Étape 5 : rendu de chaque .spz relu (mêmes vues) -> 5_export/check/<variante>.json ; arrêt si le
PSNR de test baisse de plus de quality.max_psnr_drop_db[variante] par rapport à l'entraînement.
La planche 4 vues (variante normale) est rapatriée et jointe à l'e-mail.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .context import Context, StepError

log = logging.getLogger("pipeline")
PKG_PARENT = Path(__file__).resolve().parents[1]
TRAIN_TAG = "train"


def enabled(ctx: Context) -> bool:
    return bool(ctx.cfg["quality"]["enabled"])


def command(ctx: Context, ply: Path, tag: str, out: Path, data_dir: Path, scale: float = 1.0,
            reference: Path | None = None, sheet_eye: list[float] | None = None,
            sheet_scale: float = 1.0) -> list[str]:
    q, t = ctx.cfg["quality"], ctx.cfg["train"]
    cmd = [sys.executable, "-m", "pipeline.render_check", "--ply", str(ply), "--tag", tag, "--out", str(out),
           "--data-dir", str(data_dir), "--trainer", str(t["trainer"]), "--factor", str(t["data_factor"]),
           "--test-every", str(t["test_every"]), "--train-views", str(q["train_views"]),
           "--scale=" + repr(float(scale))]
    if reference is not None:
        cmd += ["--reference", str(reference)]
    if sheet_eye is not None:
        # Forme « --opt=valeur » : une coordonnée négative (-0.8,...) serait prise pour une option.
        cmd += ["--sheet-eye=" + ",".join(f"{v:.4f}" for v in sheet_eye), "--sheet-scale", repr(float(sheet_scale)),
                "--sheet-size", str(q["sheet_size"]), "--sheet-fov", str(q["sheet_fov_deg"])]
    return cmd


def run_check(ctx: Context, cmd: list[str], out: Path, tag: str) -> dict | None:
    """Résultat du contrôle, ou None (dry-run, ou outil en échec : signalé mais non bloquant,
    pour ne pas perdre un entraînement payé à cause du contrôle lui-même)."""
    if not ctx.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    try:
        ctx.run(cmd, cwd=PKG_PARENT)
        if ctx.dry_run:
            return None
        return json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
    except (StepError, OSError, ValueError) as exc:
        log.error("Contrôle visuel « %s » impossible : %s", tag, exc)
        return {"error": str(exc)[:500]}


def summary(res: dict) -> dict:
    """Résumé pour state.json / e-mail."""
    if "error" in res:
        return {"error": res["error"]}
    out = {"test": res.get("test"), "train_views": res.get("train_views"),
           "gaussians": res.get("gaussians"), "sh_degree": res.get("sh_degree")}
    if res.get("sheet"):
        out["sheet"] = res["sheet"]
    return out


def export_drop_reason(train: dict, export: dict, variant: str, max_drop: float) -> tuple[float | None, str | None]:
    """Baisse de PSNR de test entre le rendu entraîné et le .spz ; raison d'arrêt si > max_drop."""
    a = (train.get("test") or {}).get("psnr")
    b = (export.get("test") or {}).get("psnr")
    if a is None or b is None:
        # Pas de vue de test (test_every <= 0) : on se rabat sur les vues d'entraînement choisies.
        a = (train.get("train_views") or {}).get("psnr")
        b = (export.get("train_views") or {}).get("psnr")
    if a is None or b is None:
        return None, None
    drop = round(a - b, 3)
    if drop > max_drop:
        return drop, (f"le .spz {variant} dégrade le rendu : PSNR {b:.2f} dB contre {a:.2f} dB à l'entraînement "
                      f"(baisse {drop:.2f} dB > {max_drop} dB autorisés)")
    return drop, None
