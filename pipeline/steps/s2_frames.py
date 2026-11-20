"""Étape 2 — extraction des images les plus nettes avec sharp-frames."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..context import Context, StepError
from . import s1_fetch

NAME = "frames"
log = logging.getLogger("pipeline")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def images_dir(ctx: Context) -> Path:
    return ctx.root / "2_frames" / "images"


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def outputs(ctx: Context) -> list[Path]:
    return [images_dir(ctx)]


def build_command(cfg: dict, video: Path | str, out: Path) -> list[str]:
    f = cfg["frames"]
    cmd = [
        "sharp-frames", str(video), str(out),
        "--selection-method", f["selection_method"],
        "--fps", str(f["fps"]),
        "--num-frames", str(f["num_frames"]),
        "--min-buffer", str(f["min_buffer"]),
        "--format", f["format"],
        "--force-overwrite",
    ]
    if f.get("width"):
        cmd += ["--width", str(f["width"])]
    return cmd


def run(ctx: Context) -> dict:
    video = s1_fetch.video_path(ctx) or (ctx.root / "1_input" / "video.mp4")
    out = images_dir(ctx)
    if out.exists() and not ctx.dry_run:
        shutil.rmtree(out)
    ctx.dir("2_frames")
    ctx.run(build_command(ctx.cfg, video, out))
    if ctx.dry_run:
        return {}
    return {"frames": len(list_images(out))}


def verify(ctx: Context, metrics: dict) -> None:
    imgs = list_images(images_dir(ctx))
    n, f = len(imgs), ctx.cfg["frames"]
    if n < f["min_frames"]:
        raise StepError(
            f"seulement {n} images nettes (minimum {f['min_frames']}) : vidéo trop courte ou trop floue. "
            "Augmentez frames.fps ou refilmez plus lentement."
        )
    if n < f["num_frames"]:
        log.warning("%d images retenues au lieu de %d (vidéo courte ?)", n, f["num_frames"])
    tiny = [p.name for p in imgs if p.stat().st_size < 10_000]
    if tiny:
        raise StepError(f"{len(tiny)} images quasi vides, ex. {tiny[:3]}")
