"""Étape 1 — récupération de la vidéo (chemin local ou URL HTTPS)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..context import Context, StepError

NAME = "fetch"
log = logging.getLogger("pipeline")

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
MIN_BYTES = 1_000_000  # < 1 Mo : sûrement pas une vidéo de logement


def video_path(ctx: Context) -> Path | None:
    d = ctx.root / "1_input"
    if not d.is_dir():
        return None
    files = sorted(p for p in d.iterdir() if p.name.startswith("video."))
    return files[0] if files else None


def outputs(ctx: Context) -> list[Path]:
    p = video_path(ctx)
    return [p] if p else [ctx.root / "1_input" / "video.*"]


def _ext_from(name: str) -> str:
    ext = Path(name).suffix.lower()
    return ext if ext in VIDEO_EXTS else ".mp4"


def run(ctx: Context) -> dict:
    src = ctx.input
    if not src:
        raise StepError("--input est obligatoire pour l'étape 1 (chemin local ou URL https://)")
    out_dir = ctx.dir("1_input")
    parsed = urlparse(src)

    if parsed.scheme in ("http", "https"):
        if parsed.scheme != "https":
            raise StepError("seules les URL https:// sont acceptées")
        dest = out_dir / f"video{_ext_from(parsed.path)}"
        log.info("Téléchargement de %s://%s%s -> %s", parsed.scheme, parsed.netloc, parsed.path, dest)
        if ctx.dry_run:
            return {"input_bytes": None}
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(src, stream=True, timeout=(15, 120)) as r:
            if r.status_code != 200:
                raise StepError(f"téléchargement impossible : HTTP {r.status_code}")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        tmp.replace(dest)
    else:
        p = Path(src)
        dest = out_dir / f"video{_ext_from(p.name)}"
        log.info("Copie de %s -> %s", p, dest)
        if ctx.dry_run:
            if not p.is_file():
                log.warning("[dry-run] vidéo introuvable : %s", p)
            return {"input_bytes": None}
        if not p.is_file():
            raise StepError(f"vidéo introuvable : {p}")
        if p.resolve() != dest.resolve():
            shutil.copyfile(p, dest)

    for other in out_dir.glob("video.*"):
        if other != dest:
            other.unlink()
    return {"input_bytes": dest.stat().st_size}


def verify(ctx: Context, metrics: dict) -> None:
    p = video_path(ctx)
    if p is None:
        raise StepError("aucune vidéo dans 1_input/")
    if p.stat().st_size < MIN_BYTES:
        raise StepError(f"vidéo trop petite ({p.stat().st_size} octets) : fichier corrompu ou mauvais lien ?")
    with open(p, "rb") as f:
        head = f.read(64)
    if head[:5].lower() in (b"<!doc", b"<html"):
        raise StepError("le lien a renvoyé une page HTML, pas une vidéo (lien de partage non direct ?)")
