"""Étape 5 — nettoyage, alignement et export .spz (normal + mobile) avec splat-transform.

Comportements mesurés sur @playcanvas/splat-transform v3.12 :
  - -V opacity,gt,X compare l'opacité LINÉAIRE (0..1), même si le .ply la stocke en logit :
    on passe donc min_opacity tel quel (un seuil en logit garderait tout) ;
  - --decimate doit être la dernière action ET écrire un .ply : export en deux passes ;
  - -r/-s/-t : voir alignment.splat_transform_args.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .. import quality
from ..alignment import Alignment, apply_overrides, splat_transform_args
from ..context import Context, StepError
from ..plyutil import read_header, read_vertex_columns
from . import s3_colmap, s4_train

NAME = "export"
log = logging.getLogger("pipeline")


def export_dir(ctx: Context) -> Path:
    return ctx.root / "5_export"


def cleaned_path(ctx: Context) -> Path:
    return export_dir(ctx) / "cleaned.ply"


def spz_path(ctx: Context, variant: str) -> Path:
    suffix = ctx.cfg["export"]["variants"][variant]["suffix"]
    return export_dir(ctx) / f"{ctx.slug}{suffix}.spz"


def outputs(ctx: Context) -> list[Path]:
    return [cleaned_path(ctx), *[spz_path(ctx, v) for v in ctx.cfg["export"]["variants"]]]


def load_alignment(ctx: Context) -> Alignment:
    """Alignement estimé à l'étape 3, corrigé par alignment.override de la config courante."""
    import dataclasses

    data = json.loads(s3_colmap.alignment_path(ctx).read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Alignment)}
    al = Alignment(**{k: v for k, v in data.items() if k in known})  # tolère d'anciens alignment.json
    return _with_overrides(ctx, al)


def _with_overrides(ctx: Context, al: Alignment) -> Alignment:
    o = ctx.cfg["alignment"]["override"]
    if any(v is not None for v in o.values()):
        al = apply_overrides(al, o["scale_factor"], o["yaw_deg"], o["floor_offset_m"])
        log.info("Corrections manuelles appliquées : %s", al.overrides)
    return al


def check_dir(ctx: Context) -> Path:
    return export_dir(ctx) / "check"


def train_scale(ctx: Context) -> float:
    return float(json.loads(s3_colmap.train_transform_path(ctx).read_text(encoding="utf-8"))["train_scale"])


def sheet_eye(ctx: Context) -> list[float]:
    """Point de départ de la visite (même calcul que tour-entry.ts), en mètres."""
    import numpy as np
    from .s7_tour_entry import compute_layout
    d = read_vertex_columns(cleaned_path(ctx), ["x", "y", "z"])
    return compute_layout(np.stack([d["x"], d["y"], d["z"]], 1), ctx.cfg)["rooms"][0]["position"]


def quality_check(ctx: Context, metrics: dict) -> None:
    """Rend chaque .spz relu avec les caméras de l'entraînement ; arrêt si le rendu se dégrade."""
    st = ctx.cfg["export"]["splat_transform"]
    cd = check_dir(ctx)
    if not ctx.dry_run:
        if cd.exists():
            shutil.rmtree(cd)
        cd.mkdir(parents=True)
    train_json = s4_train.check_dir(ctx) / f"{quality.TRAIN_TAG}.json"
    train_res = None
    if not ctx.dry_run:
        if not train_json.is_file():
            log.warning("Contrôle visuel de l'étape 4 absent (%s) : pas de comparaison avant/après export", train_json)
        else:
            train_res = json.loads(train_json.read_text(encoding="utf-8"))
            if "error" in train_res:
                train_res = None
    k = 1.0 if ctx.dry_run else train_scale(ctx)
    eye = [0.0, 1.5, 0.0] if ctx.dry_run else sheet_eye(ctx)
    reasons = []
    for i, name in enumerate(ctx.cfg["export"]["variants"]):
        decoded = cd / f"{name}_decoded.ply"
        try:
            ctx.run([st, "-q", "-w", spz_path(ctx, name), decoded])
        except StepError as exc:
            log.error("Contrôle visuel %s : relecture du .spz impossible (%s)", name, exc)
            metrics[f"quality_{name}"] = {"error": str(exc)[:500]}
            continue
        ref = s4_train.check_renders(ctx)
        cmd = quality.command(ctx, decoded, name, cd, s3_colmap.train_data_dir(ctx), scale=k,
                              reference=ref if train_res else None,
                              sheet_eye=eye if i == 0 else None, sheet_scale=k)
        res = quality.run_check(ctx, cmd, cd, name)
        if not ctx.dry_run:
            decoded.unlink(missing_ok=True)
        if res is None:
            continue
        q = quality.summary(res)
        if train_res and "error" not in res:
            drop, reason = quality.export_drop_reason(train_res, res, name,
                                                      ctx.cfg["quality"]["max_psnr_drop_db"][name])
            q["psnr_drop_db"] = drop
            q["psnr_vs_train_render"] = (res.get("test") or {}).get("psnr_vs_ref")
            log.info("  contrôle %s : test %s ; baisse de PSNR %s dB ; PSNR / rendu entraîné %s dB",
                     name, res.get("test"), drop, q["psnr_vs_train_render"])
            if reason:
                reasons.append(reason)
        metrics[f"quality_{name}"] = q
        if res.get("sheet"):
            metrics["quality_sheet"] = str(cd / res["sheet"])
    if reasons:
        raise StepError(" ; ".join(reasons))


def export_transform(ctx: Context) -> Alignment:
    """Transformation appliquée au .ply entraîné. L'entraînement a lieu dans le repère du site
    x train_scale (étape 3) : la rotation, le sol et le centrage sont déjà faits ; il reste à
    revenir en mètres (échelle 1/train_scale) puis à appliquer les corrections manuelles."""
    import numpy as np
    k = json.loads(s3_colmap.train_transform_path(ctx).read_text(encoding="utf-8"))["train_scale"]
    base = json.loads(s3_colmap.alignment_path(ctx).read_text(encoding="utf-8"))
    al = Alignment(scale=1.0 / k, R=np.eye(3).tolist(), t=[0.0, 0.0, 0.0],
                   up_consistency=base.get("up_consistency", 0.0),
                   camera_height_model=base.get("camera_height_model", 0.0),
                   yaw_deg=0.0, floor_model=0.0)
    return _with_overrides(ctx, al)


def clean_command(cfg: dict, src: Path, dst: Path, al: Alignment | None) -> list[str]:
    e = cfg["export"]
    cmd = [e["splat_transform"], "-q", "-w", str(src), "--filter-nan",
           "-V", f"opacity,gt,{e['min_opacity']}"]
    if al is not None:
        cmd += splat_transform_args(al)
    if e["crop_box"]:
        cmd += ["-B", ",".join(str(v) for v in e["crop_box"])]
    return cmd + [str(dst)]


def spz_info(ctx: Context, path: Path) -> dict:
    out = ctx.capture([ctx.cfg["export"]["splat_transform"], "-q", str(path), "--info", "json", "null"])
    try:
        return json.loads(out[out.index("{"): out.rindex("}") + 1])
    except ValueError as exc:
        raise StepError(f"lecture de {path.name} impossible :\n{out[-500:]}") from exc


def _export_variant(ctx: Context, name: str, v: dict, n_clean: int) -> dict:
    st = ctx.cfg["export"]["splat_transform"]
    d = export_dir(ctx)
    out = spz_path(ctx, name)
    src = cleaned_path(ctx)
    target = v["max_splats"]

    def decimate(n: int) -> Path:
        dec = d / f"_decimated_{name}.ply"
        ctx.run([st, "-q", "-w", src, "--decimate", str(n), dec])
        return dec

    base = decimate(target) if n_clean > target or ctx.dry_run else src
    n_base = min(n_clean, target)
    for attempt in range(2):
        for sh in range(v["sh_bands"], -1, -1):
            ctx.run([st, "-q", "-w", base, "--filter-harmonics", str(sh), out])
            if ctx.dry_run:
                return {}
            size_mb = out.stat().st_size / 1e6
            log.info("  %s : %d splats, SH %d -> %.1f Mo (max %s Mo)", out.name, n_base, sh, size_mb, v["max_mb"])
            if size_mb <= v["max_mb"]:
                return {"splats": n_base, "sh_bands": sh, "size_mb": round(size_mb, 2)}
        if attempt == 0:
            # Même sans harmoniques c'est trop gros : décimation proportionnelle, une fois.
            n_base = int(n_base * v["max_mb"] / size_mb * 0.95)
            log.warning("  %s trop gros sans SH : décimation à %d splats", out.name, n_base)
            base = decimate(n_base)
    raise StepError(f"{out.name} dépasse {v['max_mb']} Mo même décimé")


def run(ctx: Context) -> dict:
    ctx.dir("5_export")
    src = s4_train.splats_path(ctx)
    al = None if ctx.dry_run else export_transform(ctx)
    if ctx.dry_run:
        log.info("[dry-run] alignement lu depuis %s (rotation/échelle/translation)", s3_colmap.alignment_path(ctx))
    ctx.run(clean_command(ctx.cfg, src, cleaned_path(ctx), al))

    metrics: dict = {}
    n_clean = 0
    if al is not None:
        full = load_alignment(ctx)
        metrics["alignment_applied"] = {
            "scale_m_per_unit": round(full.scale, 6), "yaw_deg": round(full.yaw_deg, 2),
            "export_scale": round(al.scale, 6),
            "overrides": al.overrides or {"scale_factor": 1.0, "yaw_deg": 0.0, "floor_offset_m": 0.0},
        }
    if not ctx.dry_run:
        n_in = read_header(src).count()
        n_clean = read_header(cleaned_path(ctx)).count()
        log.info("Nettoyage : %d -> %d gaussiennes (%.1f %% retirées)", n_in, n_clean, 100 * (1 - n_clean / n_in))
        metrics["gaussians_cleaned"] = n_clean

    for name, v in ctx.cfg["export"]["variants"].items():
        r = _export_variant(ctx, name, v, n_clean)
        for k, val in r.items():
            metrics[f"{name}_{k}"] = val

    if not ctx.dry_run:
        for p in export_dir(ctx).glob("_decimated_*.ply"):
            p.unlink()
    if quality.enabled(ctx):
        quality_check(ctx, metrics)
    return metrics


def verify(ctx: Context, metrics: dict) -> None:
    for name, v in ctx.cfg["export"]["variants"].items():
        p = spz_path(ctx, name)
        if not p.is_file():
            raise StepError(f"{p.name} absent")
        size_mb = p.stat().st_size / 1e6
        if size_mb > v["max_mb"]:
            raise StepError(f"{p.name} : {size_mb:.1f} Mo > {v['max_mb']} Mo")
        info = spz_info(ctx, p)
        n = int(info.get("numGaussians", 0))
        if info.get("format") != "spz" or n == 0:
            raise StepError(f"{p.name} illisible ou vide : {info}")
        if n > v["max_splats"]:
            raise StepError(f"{p.name} : {n} splats > {v['max_splats']}")
        metrics[f"{name}_splats"] = n
        log.info("  vérifié : %s, %d splats, SH %s, %.1f Mo", p.name, n, info.get("shBands"), size_mb)
