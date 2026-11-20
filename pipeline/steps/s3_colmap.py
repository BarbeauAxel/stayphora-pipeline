"""Étape 3 — poses caméra avec COLMAP 4.x (global mapper, repli incrémental) + alignement.

feature_extractor -> sequential_matcher -> view_graph_calibrator (sur une COPIE de la base)
-> global_mapper. Si < global_min_ratio d'images alignées : mapper incrémental sur la base
d'origine. Si le meilleur modèle reste < min_ratio : arrêt.
Puis : alignement (repère du viewer, cf. alignment.py) et image_undistorter (PINHOLE pour gsplat).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from .. import colmap_text
from ..alignment import compute_alignment
from ..context import Context, StepError
from . import s2_frames

NAME = "colmap"
log = logging.getLogger("pipeline")


def sfm_dir(ctx: Context) -> Path:
    return ctx.root / "3_sfm"


def undistorted_dir(ctx: Context) -> Path:
    return sfm_dir(ctx) / "undistorted"


def alignment_path(ctx: Context) -> Path:
    return sfm_dir(ctx) / "alignment.json"


def train_data_dir(ctx: Context) -> Path:
    """Données d'entraînement gsplat : images non distordues + modèle dans le repère d'entraînement."""
    return sfm_dir(ctx) / "train"


def train_transform_path(ctx: Context) -> Path:
    return train_data_dir(ctx) / "transform.json"


def outputs(ctx: Context) -> list[Path]:
    u = undistorted_dir(ctx)
    t = train_data_dir(ctx)
    return [u / "images", u / "sparse", alignment_path(ctx), t / "images", t / "sparse" / "0" / "images.bin",
            train_transform_path(ctx)]


def train_scale(al, images) -> float:
    """Facteur k du repère d'entraînement = repère du site (mètres) x k, choisi pour que la
    distance médiane des caméras à leur centre vaille 1, comme la normalisation de gsplat.
    Indispensable : le bruit MCMC de gsplat (inject_noise_to_position) croît comme le CARRÉ de
    l'échelle de la scène ; dans les unités brutes de COLMAP (scene scale 840 sur playroom), il
    disperse toutes les gaussiennes et l'entraînement s'effondre (essai RunPod du 20/11/2026)."""
    centers, _ = colmap_text.camera_centers_and_up(images)
    c = al.apply(centers)
    r = float(np.median(np.linalg.norm(c - np.median(c, axis=0), axis=1)))
    if r <= 0:
        raise StepError("caméras toutes au même endroit : échelle d'entraînement impossible")
    return 1.0 / r


def _link_images(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    try:
        dst.symlink_to(Path("..") / src.parent.name / src.name, target_is_directory=True)
    except OSError:  # Windows sans droit de lien symbolique
        shutil.copytree(src, dst)


def _verify_train_bin(ctx: Context, txt: Path, sparse: Path) -> None:
    """Aller-retour : le binaire écrit à la main, relu par COLMAP, doit redonner le TXT écrit."""
    colmap = ctx.cfg["colmap"]["binary"]
    back = sfm_dir(ctx) / "_analyze" / "train_roundtrip"
    if back.exists():
        shutil.rmtree(back)
    back.mkdir(parents=True)
    out = ctx.capture([colmap, "model_analyzer", "--path", sparse])
    summary = [l.split("]")[-1].strip() for l in out.splitlines()
               if any(k in l for k in ("Cameras:", "Registered images:", "Points:", "Observations:"))]
    log.info("  model_analyzer (modèle d'entraînement) : %s", " ; ".join(summary) or out[-300:])
    ctx.run([colmap, "model_converter", "--input_path", sparse, "--output_path", back, "--output_type", "TXT"])
    problems = colmap_text.compare_models_txt(txt, back)
    if problems:
        raise StepError("modèle d'entraînement binaire incohérent après relecture par COLMAP : "
                        + " ; ".join(problems))
    log.info("  aller-retour binaire -> COLMAP -> TXT : caméras, poses et points identiques")


def _write_train_data(ctx: Context, al, k: float, keep: set[str]) -> None:
    """Modèle non distordu réécrit dans le repère d'entraînement X_train = k * (s R X + t)."""
    t = train_data_dir(ctx)
    sparse, txt = t / "sparse" / "0", t / "sparse_txt"
    for d in (sparse, txt):
        if d.exists():
            shutil.rmtree(d)
    colmap_text.transform_model_txt(sfm_dir(ctx) / "undistorted_txt", txt, k * al.scale,
                                    np.array(al.R), k * np.array(al.t), keep=keep)
    colmap_text.write_bin_for_gsplat(txt, sparse)
    _verify_train_bin(ctx, txt, sparse)
    _link_images(undistorted_dir(ctx) / "images", t / "images")
    train_transform_path(ctx).write_text(json.dumps({
        "train_scale": k,
        "note": "repère d'entraînement = repère du site (mètres, Y haut, sol à 0) x train_scale ; "
                "l'export applique l'échelle 1/train_scale puis les corrections manuelles",
    }, indent=2), encoding="utf-8")
    log.info("Modèle d'entraînement : repère du site x %.4f (1 unité = %.3f m), %s", k, 1 / k, sparse)


# ── Lecture des modèles ─────────────────────────────────────────────────────

def _is_model(d: Path) -> bool:
    return any((d / f"images.{ext}").is_file() for ext in ("bin", "txt"))


def candidate_models(out: Path) -> list[Path]:
    if not out.is_dir():
        return []
    if _is_model(out):
        return [out]
    return sorted(d for d in out.iterdir() if d.is_dir() and _is_model(d))


def usable(ctx: Context, images: colmap_text.Images) -> np.ndarray:
    c = ctx.cfg["colmap"]
    return colmap_text.usable_mask(images, c["min_image_points3d"], c["max_camera_distance_ratio"])


def model_images(ctx: Context, model: Path) -> colmap_text.Images:
    """Images recalées d'un modèle (via une conversion TXT temporaire)."""
    tmp = sfm_dir(ctx) / "_analyze" / f"{model.parent.name}_{model.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ctx.run([ctx.cfg["colmap"]["binary"], "model_converter", "--input_path", model, "--output_path", tmp,
             "--output_type", "TXT"])
    return colmap_text.read_images(tmp / "images.txt")


def best_model(ctx: Context, out: Path) -> tuple[Path | None, int]:
    """Modèle avec le plus d'images EXPLOITABLES (cf. colmap_text.usable_mask)."""
    best, best_n = None, 0
    for m in candidate_models(out):
        imgs = model_images(ctx, m)
        ok = usable(ctx, imgs)
        n = int(ok.sum())
        log.info("  modèle %s : %d images recalées, dont %d exploitables (%d écartées : 0-%d points 3D "
                 "ou pose aberrante)", m.name, len(imgs.names), n, len(imgs.names) - n,
                 ctx.cfg["colmap"]["min_image_points3d"] - 1)
        if n > best_n:
            best, best_n = m, n
    return best, best_n


# ── Étape ───────────────────────────────────────────────────────────────────

def run(ctx: Context) -> dict:
    c = ctx.cfg["colmap"]
    colmap = c["binary"]
    gpu = "1" if c["use_gpu"] else "0"
    images = s2_frames.images_dir(ctx)
    n_frames = len(s2_frames.list_images(images))
    d = sfm_dir(ctx)
    if d.exists() and not ctx.dry_run:
        shutil.rmtree(d)
    ctx.dir("3_sfm")
    db, db_global = d / "database.db", d / "database_global.db"

    ctx.run([colmap, "feature_extractor",
             "--database_path", db, "--image_path", images,
             "--ImageReader.single_camera", "1",
             "--ImageReader.camera_model", c["camera_model"],
             "--FeatureExtraction.use_gpu", gpu])

    match = [colmap, "sequential_matcher", "--database_path", db,
             "--SequentialMatching.overlap", str(c["sequential_overlap"]),
             "--FeatureMatching.use_gpu", gpu]
    if c["loop_detection"]:
        match += ["--SequentialMatching.loop_detection", "1",
                  "--SequentialMatching.vocab_tree_path", c["vocab_tree_path"]]
    ctx.run(match)

    # view_graph_calibrator modifie la base en place : on travaille sur une copie.
    log.info("Copie %s -> %s", db.name, db_global.name)
    if not ctx.dry_run:
        shutil.copyfile(db, db_global)
    code = ctx.run([colmap, "view_graph_calibrator", "--database_path", db_global], check=False)
    if code != 0:
        log.warning("view_graph_calibrator a échoué (code %d) : global_mapper sans calibration", code)
        shutil.copyfile(db, db_global)

    out_global = d / "global"
    ctx.dir("3_sfm/global")
    code = ctx.run([colmap, "global_mapper", "--database_path", db_global,
                    "--image_path", images, "--output_path", out_global], check=False)

    out_inc = d / "incremental"
    inc_cmd = [colmap, "mapper", "--database_path", db, "--image_path", images,
               "--output_path", out_inc]
    if ctx.dry_run:
        log.info("[dry-run] si < %d %% d'images alignées, repli :", round(c["global_min_ratio"] * 100))
        ctx.run(inc_cmd)
        _postprocess(ctx, d / "incremental/0", dry=True)
        return {}

    if code != 0:
        log.warning("global_mapper a échoué (code %d)", code)
    model, n = best_model(ctx, out_global)
    mapper, ratio = "global", n / max(n_frames, 1)
    log.info("global_mapper : %d/%d images alignées exploitables (%.1f %%)", n, n_frames, 100 * ratio)

    if ratio < c["global_min_ratio"]:
        log.warning("< %d %% : relance avec le mapper incrémental", round(c["global_min_ratio"] * 100))
        ctx.dir("3_sfm/incremental")
        code = ctx.run(inc_cmd, check=False)
        if code != 0:
            log.warning("mapper incrémental a échoué (code %d)", code)
        m2, n2 = best_model(ctx, out_inc)
        r2 = n2 / max(n_frames, 1)
        log.info("mapper incrémental : %d/%d images alignées exploitables (%.1f %%)", n2, n_frames, 100 * r2)
        if n2 > n:
            model, n, ratio, mapper = m2, n2, r2, "incremental"

    if model is None or ratio < c["min_ratio"]:
        raise StepError(
            f"seulement {n}/{n_frames} images alignées ({100 * ratio:.0f} % < {100 * c['min_ratio']:.0f} %). "
            "La vidéo ne permet pas une reconstruction fiable : refilmez plus lentement, "
            "avec plus de recouvrement entre les pièces et sans zones blanches/vitrées en gros plan."
        )
    log.info("Modèle retenu : %s (%s)", model, mapper)
    metrics = {"frames_total": n_frames, "aligned_images": n, "aligned_ratio": round(ratio, 4),
               "mapper": mapper}
    metrics.update(_postprocess(ctx, model))
    return metrics


def _postprocess(ctx: Context, model: Path, dry: bool = False) -> dict:
    """Alignement + image_undistorter."""
    colmap = ctx.cfg["colmap"]["binary"]
    d = sfm_dir(ctx)
    txt = d / "model_txt"
    ctx.dir("3_sfm/model_txt")
    ctx.run([colmap, "model_converter", "--input_path", model, "--output_path", txt,
             "--output_type", "TXT"])
    und = undistorted_dir(ctx)
    ctx.run([colmap, "image_undistorter", "--image_path", s2_frames.images_dir(ctx),
             "--input_path", model, "--output_path", und, "--output_type", "COLMAP"])
    und_txt = d / "undistorted_txt"
    ctx.dir("3_sfm/undistorted_txt")
    ctx.run([colmap, "model_converter", "--input_path", und / "sparse", "--output_path", und_txt,
             "--output_type", "TXT"])
    if dry:
        log.info("[dry-run] alignement calculé depuis %s -> %s", txt, alignment_path(ctx))
        log.info("[dry-run] modèle d'entraînement (repère du site normalisé) -> %s", train_data_dir(ctx))
        return {}

    a = ctx.cfg["alignment"]
    all_imgs = colmap_text.read_images(txt / "images.txt")
    ok = usable(ctx, all_imgs)
    imgs = all_imgs.subset(ok)
    rejected = [n for n, k in zip(all_imgs.names, ok) if not k]
    if rejected:
        log.info("Images écartées (pas assez de points 3D ou pose aberrante) : %d — %s", len(rejected),
                 ", ".join(rejected[:20]) + (" ..." if len(rejected) > 20 else ""))
    try:
        al = compute_alignment(imgs,
                               colmap_text.read_points(txt / "points3D.txt"),
                               camera_height_m=a["camera_height_m"],
                               floor_percentile=a["floor_percentile"],
                               align_walls=a["align_walls"])
    except ValueError as exc:
        raise StepError(f"alignement impossible : {exc}") from exc
    alignment_path(ctx).write_text(json.dumps(al.to_dict(), indent=2), encoding="utf-8")
    k = train_scale(al, imgs)
    _write_train_data(ctx, al, k, set(imgs.names))
    log.info("Alignement : échelle %.4f m/unité, cap %.0f°, cohérence verticale %.2f",
             al.scale, al.yaw_deg, al.up_consistency)
    if al.up_consistency < a["min_up_consistency"]:
        log.warning("Verticale peu fiable (%.2f < %.2f) : téléphone penché ou vidéo tournée ? "
                    "Vérifiez l'orientation de la scène avec /demo/<slug>?debug.",
                    al.up_consistency, a["min_up_consistency"])
    log.info("  verticale à %.1f° de l'axe -Y COLMAP ; roulis caméras médian %.1f° (p90 %.1f°), "
             "tangage médian %.1f° ; orientations %s ; hauteur caméra supposée %.2f m",
             al.vertical_angle_deg, al.camera_roll_median_deg, al.camera_roll_p90_deg,
             al.camera_pitch_median_deg, al.rotated_images, al.camera_height_assumed_m)
    if al.camera_roll_p90_deg > 15:
        log.warning("Roulis élevé (p90 %.0f°) : téléphone penché sur le côté ? Vérifiez l'horizontalité "
                    "de la scène avec /demo/<slug>?debug.", al.camera_roll_p90_deg)
    fc = al.floor_check or {}
    if "tilt_deg" in fc:
        log.info("  contrôle du sol (nuage de points) : plan à %.2f° de l'horizontale, à y = %.3f m, "
                 "%d points (cible < %.0f°) -> %s", fc["tilt_deg"], fc["plane_height_m"], fc["inliers"],
                 fc["target_deg"], fc["status"])
        if fc["tilt_deg"] > fc["warn_deg"]:
            log.warning("Sol incliné de %.1f° (> %.0f°) : verticale estimée par les caméras douteuse. "
                        "Vérifiez avec /demo/<slug>?debug.", fc["tilt_deg"], fc["warn_deg"])
    else:
        log.warning("Contrôle du sol impossible : %s", fc.get("reason", "?"))
    return {"alignment_estimated": al.diagnostics(), "registered_images": len(all_imgs.names),
            "images_rejected": rejected, "train_scale": round(k, 6)}


def verify(ctx: Context, metrics: dict) -> None:
    u = undistorted_dir(ctx)
    imgs = s2_frames.list_images(u / "images")
    if len(imgs) < metrics.get("aligned_images", 1) * 0.95:
        raise StepError(f"image_undistorter n'a produit que {len(imgs)} images")
    sparse = u / "sparse"
    if not (_is_model(sparse) or _is_model(sparse / "0")):
        raise StepError(f"modèle non distordu absent de {sparse}")
    if not alignment_path(ctx).is_file():
        raise StepError("alignment.json absent")
    if metrics.get("aligned_ratio", 0) < ctx.cfg["colmap"]["min_ratio"]:
        raise StepError("taux d'images alignées sous le seuil")
