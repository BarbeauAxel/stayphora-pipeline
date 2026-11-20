"""python -m pipeline run --input <video|url> --slug <slug> [--config config.yaml] ..."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigError, default_work_dir, load_config, validate_slug
from .context import Context, StepError, setup_console_logging
from .envsecrets import load_dotenv
from .runner import run_pipeline
from .steps import STEPS

log = logging.getLogger("pipeline")


def _parse_steps(value: str) -> set[int]:
    try:
        out = {int(x) for x in value.split(",") if x.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError("liste de numéros d'étapes, ex. 6,8")
    if any(not 1 <= n <= len(STEPS) for n in out):
        raise argparse.ArgumentTypeError(f"étapes entre 1 et {len(STEPS)}")
    return out


def build_parser() -> argparse.ArgumentParser:
    steps_help = ", ".join(f"{i}={m.NAME}" for i, m in enumerate(STEPS, 1))
    p = argparse.ArgumentParser(prog="python -m pipeline", description="Vidéo de logement -> visite 3D .spz")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="exécute le pipeline", description=f"Étapes : {steps_help}")
    r.add_argument("--input", help="vidéo : chemin local ou URL https:// (obligatoire pour l'étape 1)")
    r.add_argument("--slug", required=True, help="identifiant de la visite (a-z, 0-9, tirets)")
    r.add_argument("--config", default=None, help="fichier YAML (défaut : ./config.yaml s'il existe)")
    r.add_argument("--from-step", type=int, default=1, metavar="N", help="reprend à l'étape N")
    r.add_argument("--to-step", type=int, default=None, metavar="N", help="s'arrête après l'étape N")
    r.add_argument("--skip", type=_parse_steps, default=set(), metavar="N,M",
                   help="étapes à ignorer, ex. --skip 6,8 (pas de R2 ni d'e-mail)")
    r.add_argument("--work-dir", default=None, help="dossier de travail (défaut : config ou automatique)")
    r.add_argument("--dry-run", action="store_true", help="affiche les commandes sans rien exécuter")
    r.add_argument("-v", "--verbose", action="store_true", help="affiche la sortie des outils")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(args.verbose or args.dry_run)
    load_dotenv(Path.cwd() / ".env")
    try:
        slug = validate_slug(args.slug)
        cfg_path = args.config or ("config.yaml" if Path("config.yaml").is_file() else None)
        cfg = load_config(cfg_path)
        n = len(STEPS)
        to_step = args.to_step or n
        if not 1 <= args.from_step <= to_step <= n:
            raise ConfigError(f"il faut 1 <= --from-step <= --to-step <= {n}")
        if args.from_step == 1 and not args.input and 1 not in args.skip:
            raise ConfigError("--input est obligatoire quand on part de l'étape 1")
        work = Path(args.work_dir or cfg["work_dir"] or default_work_dir())
        ctx = Context(slug=slug, cfg=cfg, root=work / slug, input=args.input, dry_run=args.dry_run)
        log.info("Visite « %s » — dossier %s%s", slug, ctx.root, " (DRY-RUN)" if args.dry_run else "")
        run_pipeline(ctx, from_step=args.from_step, to_step=to_step, skip=args.skip)
    except (ConfigError, StepError) as exc:
        log.error("%s", exc)
        return 2 if isinstance(exc, ConfigError) else 1
    except KeyboardInterrupt:
        return 130
    log.info("Terminé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
