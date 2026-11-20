"""Étape 7 — génère work/<slug>/tour-entry.ts : un objet `Tour` prêt à coller dans
content/fr/tours.ts du site (type défini dans content/types.ts).

Le site n'affiche que les slugs déclarés dans tours.ts : publier le .spz ne suffit pas.
Les coordonnées sont celles du repère aligné (mètres, y vers le haut, sol à 0).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np

from ..context import Context, StepError
from ..plyutil import read_vertex_columns
from . import s5_export, s6_publish

NAME = "tour_entry"
log = logging.getLogger("pipeline")


def entry_path(ctx: Context) -> Path:
    return ctx.root / "tour-entry.ts"


def outputs(ctx: Context) -> list[Path]:
    return [entry_path(ctx)]


def _r(x: float) -> float:
    v = round(float(x), 1)
    return 0.0 if v == 0 else v


def compute_layout(xyz: np.ndarray, cfg: dict) -> dict:
    t = cfg["tour_entry"]
    p = t["bounds_percentile"]
    m = t["bounds_margin_m"]
    lo = np.percentile(xyz, p, axis=0)
    hi = np.percentile(xyz, 100 - p, axis=0)
    xmin, xmax = lo[0] + m, hi[0] - m
    zmin, zmax = lo[2] + m, hi[2] - m
    if xmin >= xmax:
        xmin = xmax = (lo[0] + hi[0]) / 2
    if zmin >= zmax:
        zmin = zmax = (lo[2] + hi[2]) / 2
    cx, cz = (xmin + xmax) / 2, (zmin + zmax) / 2
    eye = t["eye_height_m"]
    ymin, ymax = t["bounds_y"]
    # Vue d'ensemble : au centre, regard vers le fond (axe le plus long = Z après alignement).
    target_z = cz + 0.8 * (zmax - cz) if zmax > cz else cz + 1.0
    return {
        "bounds": {"min": [_r(xmin), _r(ymin), _r(zmin)], "max": [_r(xmax), _r(ymax), _r(zmax)]},
        "rooms": [{"label": "Vue d'ensemble", "position": [_r(cx), _r(eye), _r(cz)],
                   "target": [_r(cx), _r(eye - 0.3), _r(target_z)]}],
        "extent_m": [round(float(hi[i] - lo[i]), 2) for i in range(3)],
    }


def _vec(v: list[float]) -> str:
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


def render_entry(slug: str, scene: str, scene_mobile: str | None, layout: dict) -> str:
    s = json.dumps  # chaînes JSON = chaînes TypeScript valides
    b, room = layout["bounds"], layout["rooms"][0]
    lines = [
        f"// Généré par stayphora-pipeline le {date.today().isoformat()} pour « {slug} ».",
        "// À coller dans le tableau `tours` de content/fr/tours.ts, puis remplacer les TODO.",
        f"// Points de vue : ouvrir /demo/{slug}?debug, se placer, copier la ligne affichée.",
        f"// Emprise mesurée (x, y, z) : {layout['extent_m']} m.",
        "  {",
        f"    slug: {s(slug)},",
        f"    title: {s('TODO titre')},",
        f"    location: {s('TODO ville')},",
        f"    kind: {s('TODO type de logement')},",
        "    // area: 45,",
        f"    description: {s('TODO description')},",
        f"    scene: {s(scene)},",
    ]
    if scene_mobile:
        lines.append(f"    sceneMobile: {s(scene_mobile)},")
    lines += [
        "    cover: {",
        f"      src: {s(f'/images/tours/{slug}.jpg')},",
        f"      alt: {s('TODO description de la capture')},",
        "    },",
        f"    // video: {{ webm: {s(f'/videos/tours/{slug}.webm')}, mp4: {s(f'/videos/tours/{slug}.mp4')} }},",
        f"    bounds: {{ min: {_vec(b['min'])}, max: {_vec(b['max'])} }},",
        "    rooms: [",
        f"      {{ label: {s(room['label'])}, position: {_vec(room['position'])}, target: {_vec(room['target'])} }},",
        "    ],",
        "  },",
        "",
    ]
    return "\n".join(lines)


def run(ctx: Context) -> dict:
    variants = ctx.cfg["export"]["variants"]
    scene = s6_publish.object_key(ctx.cfg, s5_export.spz_path(ctx, "normal").name)
    scene_mobile = (s6_publish.object_key(ctx.cfg, s5_export.spz_path(ctx, "mobile").name)
                    if "mobile" in variants else None)
    if ctx.dry_run:
        log.info("[dry-run] bounds calculés depuis %s -> %s", s5_export.cleaned_path(ctx), entry_path(ctx))
        return {}
    cols = read_vertex_columns(s5_export.cleaned_path(ctx), ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) == 0:
        raise StepError("aucune gaussienne valide dans cleaned.ply")
    layout = compute_layout(xyz, ctx.cfg)
    entry_path(ctx).write_text(render_entry(ctx.slug, scene, scene_mobile, layout), encoding="utf-8")
    log.info("tour-entry.ts écrit : bounds %s, emprise %s m", layout["bounds"], layout["extent_m"])
    return {"bounds": layout["bounds"], "extent_m": layout["extent_m"]}


def verify(ctx: Context, metrics: dict) -> None:
    text = entry_path(ctx).read_text(encoding="utf-8")
    for needle in (f'slug: "{ctx.slug}"', "scene:", "bounds:", "rooms:"):
        if needle not in text:
            raise StepError(f"tour-entry.ts incomplet ({needle})")
    b = metrics["bounds"]
    if any(lo > hi for lo, hi in zip(b["min"], b["max"])):
        raise StepError(f"bounds incohérents : {b}")
    ext = metrics["extent_m"]
    if max(ext[0], ext[2]) < 1.5 or max(ext) > 200:
        log.warning("Emprise inhabituelle %s m : vérifiez camera_height_m / l'alignement", ext)
