"""Chargement de config.yaml, fusionné sur les valeurs par défaut."""

from __future__ import annotations

import copy
import os
import platform
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "work_dir": None,
    "frames": {
        "fps": 10,
        "num_frames": 400,
        "min_buffer": 3,
        "selection_method": "best-n",
        "width": 1920,
        "format": "jpg",
        "min_frames": 100,
    },
    "colmap": {
        "binary": "colmap",
        "camera_model": "SIMPLE_RADIAL",
        "use_gpu": True,
        "sequential_overlap": 15,
        "loop_detection": False,
        "vocab_tree_path": None,
        "global_min_ratio": 0.90,
        "min_ratio": 0.70,
        "min_image_points3d": 15,
        "max_camera_distance_ratio": 10.0,
    },
    "alignment": {
        "camera_height_m": 1.5,
        "floor_percentile": 2.0,
        "min_up_consistency": 0.8,
        "align_walls": True,
        "override": {"scale_factor": None, "yaw_deg": None, "floor_offset_m": None},
    },
    "train": {
        "python": "python",
        "trainer": "/opt/gsplat/examples/simple_trainer.py",
        "cap_max": 1_000_000,
        "guard": {
            "enabled": True,
            "checkpoints": [1000, 3000],
            "loss_rise_ratio": 1.3,
            "loss_rise_min": 0.03,
            "max_dead_fraction": 0.5,
            "min_gaussians_ratio": 0.5,
            "scene_scale_range": [0.5, 10.0],
        },
        "max_steps": 30_000,
        "data_factor": 1,
        "test_every": 1000,
        "extra_args": [],
    },
    "export": {
        "splat_transform": "splat-transform",
        "min_opacity": 0.05,
        "crop_box": None,
        "variants": {
            "normal": {"suffix": "", "max_splats": 1_000_000, "max_mb": 16, "sh_bands": 2},
            "mobile": {"suffix": "-mobile", "max_splats": 500_000, "max_mb": 8, "sh_bands": 0},
        },
    },
    "publish": {
        "bucket": "stayphora-assets",
        "prefix": "tours/",
        "assets_url": "https://assets.stayphora.fr",
        "tours_url": "https://visite.stayphora.fr",
        "cache_control": "public, max-age=3600",
    },
    "tour_entry": {
        "bounds_percentile": 2.0,
        "bounds_margin_m": 0.3,
        "bounds_y": [1.0, 2.2],
        "eye_height_m": 1.6,
    },
    "notify": {
        "enabled": True,
        "to": "contact@stayphora.fr",
        "from": "Stayphora Pipeline <pipeline@stayphora.fr>",
    },
    # Lu uniquement par scripts/runpod_job.py (retiré de la config envoyée au pod).
    "runpod": {
        "gpus": [
            {"id": "NVIDIA GeForce RTX 4090", "cloud": "COMMUNITY"},
            {"id": "NVIDIA GeForce RTX 3090", "cloud": "COMMUNITY"},
            {"id": "NVIDIA GeForce RTX 5090", "cloud": "COMMUNITY"},
        ],
        "max_price_per_hour": 0.75,
        "max_hours": 3.0,
        "disk_gb": 60,
        "image": "ghcr.io/barbeauaxel/stayphora-pipeline:0.1.0",
    },
}

# Même motif que lib/tour-routing.ts du site (/^\/([a-z0-9-]+)\/?$/), en plus strict :
# pas de tiret au début, à la fin ni doublé.
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# Chemins passés tels quels sur visite.stayphora.fr (TOURS_HOST_PASSTHROUGH) ou réservés.
RESERVED_SLUGS = {"demo", "images", "videos", "licences", "icon", "apple-icon", "cdn-cgi", "_next",
                  "favicon", "robots", "manifest"}
SLUG_MAX_LEN = 60


class ConfigError(ValueError):
    pass


def validate_slug(slug: str) -> str:
    if not SLUG_RE.fullmatch(slug or ""):
        raise ConfigError(
            f"slug invalide « {slug} » : minuscules, chiffres et tirets uniquement "
            "(ex. « studio-lyon-3 »), sans tiret au début, à la fin ni doublé."
        )
    if len(slug) > SLUG_MAX_LEN:
        raise ConfigError(f"slug trop long ({len(slug)} > {SLUG_MAX_LEN} caractères).")
    if slug in RESERVED_SLUGS:
        raise ConfigError(f"slug réservé par le site : « {slug} ».")
    return slug


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in base:
            raise ConfigError(f"clé de configuration inconnue : {path}{key}")
        if isinstance(base[key], dict) and key not in ("variants",):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{key} doit être une section")
            out[key] = _merge(base[key], value, f"{path}{key}.")
        else:
            out[key] = copy.deepcopy(value)
    return out


def default_work_dir() -> Path:
    env = os.environ.get("STAYPHORA_WORK_DIR")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        return Path("C:/stayphora-work")
    return Path("/workspace/work")


def load_config(path: str | os.PathLike | None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"fichier de configuration introuvable : {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{p} doit contenir un dictionnaire YAML")
    cfg = _merge(DEFAULTS, data)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    f, c, e = cfg["frames"], cfg["colmap"], cfg["export"]
    if f["num_frames"] < f["min_frames"]:
        raise ConfigError("frames.num_frames doit être >= frames.min_frames")
    if not 0 < c["min_ratio"] <= c["global_min_ratio"] <= 1:
        raise ConfigError("il faut 0 < colmap.min_ratio <= colmap.global_min_ratio <= 1")
    if c["loop_detection"] and not c["vocab_tree_path"]:
        raise ConfigError("colmap.loop_detection exige colmap.vocab_tree_path")
    if not 0 <= e["min_opacity"] < 1:
        raise ConfigError("export.min_opacity est une opacité réelle entre 0 et 1")
    for name, v in e["variants"].items():
        for k in ("suffix", "max_splats", "max_mb", "sh_bands"):
            if k not in v:
                raise ConfigError(f"export.variants.{name}.{k} manquant")
        if v["sh_bands"] not in (0, 1, 2, 3):
            raise ConfigError(f"export.variants.{name}.sh_bands doit valoir 0..3")
    r = cfg["runpod"]
    if not r["gpus"]:
        raise ConfigError("runpod.gpus : au moins un GPU")
    for g in r["gpus"]:
        if not isinstance(g, dict) or not g.get("id") or g.get("cloud", "COMMUNITY") not in ("COMMUNITY", "SECURE"):
            raise ConfigError(f"runpod.gpus : entrée invalide {g!r} (id + cloud COMMUNITY|SECURE)")
    if not float(r["max_price_per_hour"]) > 0:
        raise ConfigError("runpod.max_price_per_hour doit être > 0")
    o = cfg["alignment"]["override"]
    if o["scale_factor"] is not None and not float(o["scale_factor"]) > 0:
        raise ConfigError("alignment.override.scale_factor doit être > 0")
    if "normal" not in e["variants"]:
        raise ConfigError("export.variants doit contenir « normal »")
    if e["crop_box"] is not None and len(e["crop_box"]) != 6:
        raise ConfigError("export.crop_box : 6 nombres [xmin,ymin,zmin,xmax,ymax,zmax]")
