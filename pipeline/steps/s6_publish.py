"""Étape 6 — publication des .spz sur Cloudflare R2 (API S3, boto3).

Clés : <prefix><slug>.spz et <prefix><slug>-mobile.spz (prefix = « tours/ », comme le site).
Servies ensuite par <assets_url>/tours/<slug>.spz.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from ..context import Context, StepError
from ..envsecrets import get_secret, missing
from . import s5_export

NAME = "publish"
log = logging.getLogger("pipeline")
R2_SECRETS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def outputs(ctx: Context) -> list[Path]:
    return []


def object_key(cfg: dict, filename: str) -> str:
    prefix = cfg["publish"]["prefix"]
    return f"{prefix.rstrip('/')}/{filename}" if prefix else filename


def public_url(cfg: dict, key: str) -> str:
    return f"{cfg['publish']['assets_url'].rstrip('/')}/{key}"


def _client():
    import boto3
    from botocore.config import Config

    account = get_secret("R2_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=get_secret("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=get_secret("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _files(ctx: Context) -> list[Path]:
    return [s5_export.spz_path(ctx, v) for v in ctx.cfg["export"]["variants"]]


def run(ctx: Context) -> dict:
    p = ctx.cfg["publish"]
    files = _files(ctx)
    if ctx.dry_run:
        for f in files:
            log.info("[dry-run] envoi %s -> r2://%s/%s", f.name, p["bucket"], object_key(ctx.cfg, f.name))
        return {}
    absent = missing(R2_SECRETS)
    if absent:
        raise StepError(f"variables d'environnement manquantes : {', '.join(absent)}")
    from botocore.exceptions import ClientError

    s3 = _client()
    urls = {}
    for f in files:
        key = object_key(ctx.cfg, f.name)
        try:
            s3.head_object(Bucket=p["bucket"], Key=key)
            log.warning("r2://%s/%s existe déjà : il est remplacé", p["bucket"], key)
        except ClientError:
            pass
        log.info("Envoi %s (%.1f Mo) -> r2://%s/%s", f.name, f.stat().st_size / 1e6, p["bucket"], key)
        s3.upload_file(str(f), p["bucket"], key, ExtraArgs={
            "ContentType": "application/octet-stream",
            "CacheControl": p["cache_control"],
        })
        urls[f.name] = public_url(ctx.cfg, key)
    return {"published_urls": urls}


def verify(ctx: Context, metrics: dict) -> None:
    p = ctx.cfg["publish"]
    s3 = _client()
    for f in _files(ctx):
        key = object_key(ctx.cfg, f.name)
        head = s3.head_object(Bucket=p["bucket"], Key=key)
        if int(head["ContentLength"]) != f.stat().st_size:
            raise StepError(f"taille différente sur R2 pour {key}")
        url = public_url(ctx.cfg, key)
        r = requests.get(url, headers={"Range": "bytes=0-15"}, timeout=30)
        if r.status_code not in (200, 206):
            raise StepError(f"{url} non accessible publiquement (HTTP {r.status_code})")
        log.info("  vérifié : %s (HTTP %d)", url, r.status_code)
