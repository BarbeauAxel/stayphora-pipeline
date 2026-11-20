"""Diagnostic d'un ou plusieurs fichiers de gaussiennes (.ply ou .spz), sans GPU.

    python scripts/inspect_splats.py cleaned.ply test-playroom.spz --out diag --eye -0.8,1.5,-0.5

Pour chaque fichier : statistiques (positions, tailles exp(log_scale), opacités, couleurs DC,
gaussiennes hors de la pièce) en JSON, et planche de 4 vues depuis --eye (repère du site,
y en haut). Un .spz est d'abord reconverti en .ply avec splat-transform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.splat_render import load_gaussians, room_sheet, splat_stats  # noqa: E402


def splat_transform() -> str:
    local = ROOT / "node_modules" / ".bin" / ("splat-transform.cmd" if sys.platform == "win32" else "splat-transform")
    return str(local) if local.exists() else (shutil.which("splat-transform") or "splat-transform")


def as_ply(path: Path, out: Path) -> Path:
    if path.suffix.lower() == ".ply":
        return path
    dst = out / f"{path.stem}_decoded.ply"
    subprocess.run([splat_transform(), "-q", "-w", str(path), str(dst)], check=True)
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("diag"))
    p.add_argument("--eye", default="-0.8,1.5,-0.5", help="position des 4 vues (x,y,z en mètres)")
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eye = [float(v) for v in a.eye.split(",")]
    for f in a.files:
        ply = as_ply(f, a.out)
        st = splat_stats(ply)
        (a.out / f"{f.stem}_stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        print(json.dumps({k: st[k] for k in ("file", "count", "scale_max_axis_m", "opacity", "outside_box")}))
        if not a.no_render:
            from PIL import Image
            png = a.out / f"{f.stem}_4vues.png"
            Image.fromarray(room_sheet(load_gaussians(ply), eye)).save(png)
            print("planche :", png)


if __name__ == "__main__":
    main()
