#!/usr/bin/env bash
# Smoke test GPU, exécuté sur le pod juste après le démarrage, AVANT tout calcul long.
# Envoyé par scripts/runpod_job.py (pas besoin de reconstruire l'image). Durée visée < 2 min.
# Chaque vérification affiche OK/ÉCHEC ; code de sortie != 0 au premier échec.
#   1. nvidia-smi                      : le GPU est visible dans le conteneur
#   2. PyTorch                         : torch.cuda.is_available() + nom du GPU + calcul
#   3. gsplat                          : import + mini rendu CUDA (rasterization) non vide
#   4. COLMAP                          : build « with CUDA » + extraction SIFT sur GPU
#   5. splat-transform, sharp-frames   : présents
set -uo pipefail
[[ -f /etc/stayphora-env.sh ]] && . /etc/stayphora-env.sh
PY=/opt/venv/bin/python
T0=$(date +%s)
fail() { echo "ÉCHEC : $1"; echo "SMOKE TEST ÉCHOUÉ après $(( $(date +%s) - T0 )) s"; exit 1; }
ok() { echo "OK     : $1"; }

echo "== 1. nvidia-smi"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader \
    || fail "nvidia-smi (GPU non visible dans le conteneur)"
ok "nvidia-smi"

echo "== 2. PyTorch CUDA"
$PY - <<'EOF' || fail "PyTorch ne voit pas CUDA"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() = False"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum().item()
assert y == y, "résultat NaN"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), GPU {name}, sm_{cap[0]}{cap[1]}")
EOF
ok "PyTorch CUDA"

echo "== 3. gsplat (mini rendu CUDA)"
$PY - <<'EOF' || fail "gsplat : import ou rendu CUDA"
import math, torch, gsplat
from gsplat import rasterization
import fused_ssim  # extension compilée utilisée par le trainer
dev = "cuda"
n = 2000
torch.manual_seed(0)
means = (torch.rand(n, 3, device=dev) - 0.5) * 2 + torch.tensor([0, 0, 4.0], device=dev)
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
scales = torch.full((n, 3), 0.05, device=dev)
opacities = torch.full((n,), 0.8, device=dev)
colors = torch.rand(n, 3, device=dev)
viewmats = torch.eye(4, device=dev)[None]
w, h, f = 256, 192, 200.0
Ks = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], device=dev)[None]
img, alpha, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, w, h)
torch.cuda.synchronize()
assert img.shape == (1, h, w, 3), img.shape
cover = (alpha > 0.01).float().mean().item()
assert torch.isfinite(img).all() and cover > 0.05, f"rendu vide (couverture {cover:.3f})"
print(f"gsplat {gsplat.__version__} : rendu {w}x{h} OK, couverture {cover:.0%}")
EOF
ok "gsplat"

echo "== 4. COLMAP (CUDA)"
head=$(colmap -h 2>&1 | head -1)
echo "$head"
[[ "$head" == *"with CUDA"* ]] || fail "COLMAP n'est pas compilé avec CUDA"
tmp=$(mktemp -d)
$PY - "$tmp/images" <<'EOF' || fail "génération des images de test"
import os, sys, numpy as np, cv2
d = sys.argv[1]; os.makedirs(d)
rng = np.random.default_rng(0)
base = (rng.random((480, 640)) * 255).astype(np.uint8)
base = cv2.GaussianBlur(cv2.resize(cv2.resize(base, (80, 60)), (640, 480)), (0, 0), 1.5)
for i in range(2):
    cv2.imwrite(f"{d}/{i}.jpg", np.roll(base, i * 15, axis=1))
EOF
colmap feature_extractor --database_path "$tmp/db.db" --image_path "$tmp/images" \
    --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1 > "$tmp/fe.log" 2>&1 \
    || { tail -20 "$tmp/fe.log"; fail "COLMAP feature_extractor sur GPU"; }
nfeat=$($PY -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute('select coalesce(sum(rows),0) from keypoints').fetchone()[0])" "$tmp/db.db")
[[ "$nfeat" -gt 0 ]] || { tail -20 "$tmp/fe.log"; fail "COLMAP : aucun point SIFT extrait"; }
rm -rf "$tmp"
ok "COLMAP CUDA (SIFT GPU : $nfeat points)"

echo "== 5. outils"
splat-transform --version || fail "splat-transform"
sharp-frames --help > /dev/null 2>&1 || fail "sharp-frames"
ok "splat-transform, sharp-frames"

echo "SMOKE TEST OK en $(( $(date +%s) - T0 )) s"
