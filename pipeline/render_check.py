"""Contrôle visuel par rendu gsplat (GPU) : exécuté en sous-processus par les étapes 4 et 5.

    python -m pipeline.render_check --ply splats.ply --tag train --out 4_train/check \\
        --data-dir 3_sfm/train --trainer /opt/gsplat/examples/simple_trainer.py --test-every 8

Rend, avec le même rasteriseur que l'entraînement (gsplat.rasterization, fond noir) :
  - toutes les vues de test (1 image sur test_every, mises de côté par le trainer) ;
  - train_views caméras d'entraînement réparties sur la trajectoire ;
et mesure PSNR / SSIM / LPIPS contre les photos. Avec --reference (dossier d'un contrôle
précédent), mesure aussi le PSNR contre les rendus de référence (rendu .spz vs rendu entraîné).
Avec --sheet-eye, ajoute une planche de 4 vues depuis ce point (repère du site, en mètres).

Sorties dans --out :
  <tag>.json                         métriques (moyennes test, vues d'entraînement, par vue)
  <tag>_<image>.png                  photo | rendu, pour les vues d'entraînement choisies
  <tag>_planche.png                  planche 4 vues (si --sheet-eye)
  renders/<tag>/<image>.png          rendus bruts (référence des contrôles suivants, non rapatriés)

--scale k : le .ply est en mètres (export) et les caméras dans le repère d'entraînement
(mètres x k) ; positions x k et log-échelles + ln k. Aucune dépendance GPU à l'import.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .plyutil import read_header, read_vertex_columns

SH_C0 = 0.28209479177387814


# ── Parties pures (testées sans GPU) ─────────────────────────────────────────

def split_indices(n: int, test_every: int) -> tuple[np.ndarray, np.ndarray]:
    """Même partage que datasets/colmap.py de gsplat 1.5.3 : test = indices % test_every == 0."""
    idx = np.arange(n)
    if test_every <= 0:
        return idx, idx[:0]
    return idx[idx % test_every != 0], idx[idx % test_every == 0]


def pick_train_views(train_idx: np.ndarray, k: int) -> list[int]:
    """k vues d'entraînement réparties sur la trajectoire (1/(2k), 3/(2k)... du parcours)."""
    if len(train_idx) == 0 or k <= 0:
        return []
    pos = [int((2 * i + 1) * len(train_idx) / (2 * k)) for i in range(k)]
    return sorted({int(train_idx[min(p, len(train_idx) - 1)]) for p in pos})


def load_splat_arrays(path: Path, scale: float = 1.0) -> dict:
    """Colonnes du .ply -> tableaux pour gsplat.rasterization (numpy, float32).

    f_rest_* du format 3DGS : canal par canal (R0..R(K-2), G..., B...) -> (N, K-1, 3)."""
    props = read_header(path).props()
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda p: int(p.split("_")[-1]))
    n_rest = len(rest) // 3
    sh_degree = int(round(math.sqrt(n_rest + 1))) - 1
    if (sh_degree + 1) ** 2 != n_rest + 1:
        raise ValueError(f"{path}: {len(rest)} coefficients f_rest incohérents")
    cols = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2",
            "opacity", "f_dc_0", "f_dc_1", "f_dc_2", *rest]
    d = read_vertex_columns(path, cols)
    n = len(d["x"])
    sh = np.zeros((n, n_rest + 1, 3), np.float32)
    sh[:, 0] = np.stack([d[f"f_dc_{i}"] for i in range(3)], 1)
    if n_rest:
        R = np.stack([d[c] for c in rest], 1).reshape(n, 3, n_rest)
        sh[:, 1:] = R.transpose(0, 2, 1)
    q = np.stack([d[f"rot_{i}"] for i in range(4)], 1)
    return {
        "means": (np.stack([d["x"], d["y"], d["z"]], 1) * scale).astype(np.float32),
        "quats": (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32),
        "scales": (np.exp(np.stack([d[f"scale_{i}"] for i in range(3)], 1)) * scale).astype(np.float32),
        "opacities": (1 / (1 + np.exp(-d["opacity"]))).astype(np.float32),
        "sh": sh,
        "sh_degree": sh_degree,
    }


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse <= 1e-10 else -10 * math.log10(mse)


def mean_metrics(views: dict, names: list[str]) -> dict:
    out = {}
    for key in ("psnr", "ssim", "lpips", "psnr_vs_ref"):
        vals = [views[n][key] for n in names if views[n].get(key) is not None]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
    out["n"] = len(names)
    return out


def sheet_cameras(eye, eye_scale: float, fov_deg: float, W: int, H: int, pitch: float = -0.15):
    """4 caméras (w2c OpenCV, K) depuis eye (mètres, y en haut), caps 0/90/180/270°."""
    eye = np.asarray(eye, float) * eye_scale
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    cams = []
    for yaw in (0, 90, 180, 270):
        fwd = np.array([math.sin(math.radians(yaw)), pitch, math.cos(math.radians(yaw))])
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(fwd, right), fwd])
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ eye
        cams.append((w2c, K))
    return cams


# ── Rendu (GPU) ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--train-views", type=int, default=3)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--reference", default=None, help="dossier des rendus d'un contrôle précédent (renders/<tag>)")
    p.add_argument("--sheet-eye", default=None, help="x,y,z en mètres (repère du site)")
    p.add_argument("--sheet-scale", type=float, default=1.0, help="mètres -> repère d'entraînement")
    p.add_argument("--sheet-size", default="640x480")
    p.add_argument("--sheet-fov", type=float, default=70.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-test", type=int, default=0, help="TEST : nombre max de vues de test (0 = toutes)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    import imageio.v2 as imageio
    import torch
    from gsplat import rasterization

    sys.path.insert(0, str(a.trainer.parent))
    from datasets.colmap import Parser  # gsplat/examples

    dev = torch.device(a.device)
    parser = Parser(data_dir=str(a.data_dir), factor=a.factor, normalize=False, test_every=a.test_every)
    names = list(parser.image_names)
    train_idx, test_idx = split_indices(len(names), a.test_every)
    chosen = pick_train_views(train_idx, a.train_views)
    if a.limit_test > 0:
        test_idx = test_idx[:a.limit_test]

    s = load_splat_arrays(a.ply, a.scale)
    t = {k: torch.from_numpy(v).to(dev) for k, v in s.items() if k != "sh_degree"}

    def draw(w2c: np.ndarray, K: np.ndarray, W: int, H: int) -> np.ndarray:
        with torch.no_grad():
            img, _, _ = rasterization(
                t["means"], t["quats"], t["scales"], t["opacities"], t["sh"],
                torch.from_numpy(w2c).float()[None].to(dev), torch.from_numpy(K).float()[None].to(dev),
                W, H, sh_degree=s["sh_degree"], near_plane=0.01, far_plane=1e10)
        return img[0].clamp(0, 1).cpu().numpy()

    ssim_fn = lpips_fn = None
    try:  # mêmes réglages que simple_trainer.py (data_range 1 ; LPIPS alex, normalize=True)
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    except Exception as exc:  # noqa: BLE001 — métrique facultative
        print(f"SSIM indisponible : {exc}")
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    except Exception as exc:  # noqa: BLE001 — métrique facultative
        print(f"LPIPS indisponible : {exc}")

    def tens(x: np.ndarray):
        return torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev)

    rdir = a.out / "renders" / a.tag
    rdir.mkdir(parents=True, exist_ok=True)
    views: dict = {}
    test_set = set(test_idx.tolist())
    for i in [*test_idx.tolist(), *chosen]:
        name = names[i]
        cam = parser.camera_ids[i]
        K = np.asarray(parser.Ks_dict[cam], np.float64)
        W, H = parser.imsize_dict[cam]
        w2c = np.linalg.inv(parser.camtoworlds[i])
        gt = imageio.imread(parser.image_paths[i])[..., :3].astype(np.float32) / 255.0
        if gt.shape[:2] != (H, W):
            raise SystemExit(f"{name}: image {gt.shape[:2]} != caméra {(H, W)}")
        img = draw(w2c, K, W, H)
        stem = Path(name).stem
        imageio.imwrite(rdir / f"{stem}.png", (img * 255 + 0.5).astype(np.uint8))
        v = {"split": "test" if i in test_set else "train", "psnr": round(psnr(img, gt), 4)}
        with torch.no_grad():
            if ssim_fn is not None:
                v["ssim"] = round(float(ssim_fn(tens(img), tens(gt))), 4)
            if lpips_fn is not None:
                v["lpips"] = round(float(lpips_fn(tens(img), tens(gt))), 4)
        if a.reference:
            ref_png = Path(a.reference) / f"{stem}.png"
            if ref_png.is_file():
                ref = imageio.imread(ref_png)[..., :3].astype(np.float32) / 255.0
                v["psnr_vs_ref"] = round(psnr((img * 255 + 0.5).astype(np.uint8) / 255.0, ref), 4)
        views[name] = v
        if i in chosen:
            pair = np.concatenate([gt, img], 1)
            imageio.imwrite(a.out / f"{a.tag}_{stem}.png", (pair * 255 + 0.5).astype(np.uint8))

    test_names = [names[i] for i in test_idx]
    result = {
        "tag": a.tag, "ply": str(a.ply), "scale": a.scale, "gaussians": len(s["means"]),
        "sh_degree": s["sh_degree"], "test": mean_metrics(views, test_names),
        "train_views": mean_metrics(views, [names[i] for i in chosen]),
        "train_view_names": [names[i] for i in chosen], "views": views,
    }
    if a.sheet_eye:
        W, H = (int(x) for x in a.sheet_size.lower().split("x"))
        eye = [float(x) for x in a.sheet_eye.split(",")]
        tiles = [(draw(w2c, K, W, H) * 255 + 0.5).astype(np.uint8)
                 for w2c, K in sheet_cameras(eye, a.sheet_scale, a.sheet_fov, W, H)]
        sheet = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
        imageio.imwrite(a.out / f"{a.tag}_planche.png", sheet)
        result["sheet"] = f"{a.tag}_planche.png"
    (a.out / f"{a.tag}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("tag", "gaussians", "sh_degree", "test", "train_views")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
