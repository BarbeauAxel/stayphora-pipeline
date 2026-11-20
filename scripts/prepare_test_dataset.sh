#!/usr/bin/env bash
# Prépare un dataset public pour tester les étapes GPU (3 à 7) sans vidéo :
# « playroom » (Deep Blending, intérieur, 225 photos), distribué par l'Inria avec
# le papier 3D Gaussian Splatting (archive tandt_db.zip, ~650 Mo).
# Les images remplacent la sortie de l'étape 2 ; les étapes 1-2 sont marquées faites.
#   /app/scripts/prepare_test_dataset.sh <slug>   puis   python -m pipeline run --slug <slug> --from-step 3
set -euo pipefail
# Une session SSH n'hérite pas de l'ENV du conteneur (PATH sans /opt/venv/bin) : on recharge
# l'environnement écrit par docker/entrypoint.sh, et on appelle le Python du venv explicitement.
[[ -f /etc/stayphora-env.sh ]] && . /etc/stayphora-env.sh
PY=/opt/venv/bin/python
slug="${1:?usage: prepare_test_dataset.sh <slug>}"
work="${STAYPHORA_WORK_DIR:-/workspace/work}/$slug"
url="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
cache=/workspace/datasets

mkdir -p "$cache"
if [[ ! -d "$cache/db/playroom/images" ]]; then
    echo "Téléchargement de $url"
    curl -fL --retry 3 -o "$cache/tandt_db.zip" "$url"
    (cd "$cache" && "$PY" -m zipfile -e tandt_db.zip . && rm tandt_db.zip)
fi
rm -rf "$work"
mkdir -p "$work/1_input" "$work/2_frames"
cp -r "$cache/db/playroom/images" "$work/2_frames/images"
# Fichier factice : l'étape 1 est considérée comme faite.
head -c 2000000 /dev/zero > "$work/1_input/video.mp4"
n=$(ls "$work/2_frames/images" | wc -l)
cd /app
"$PY" - "$work" "$n" <<'EOF'
import sys
from pathlib import Path
from pipeline.state import State
work, n = Path(sys.argv[1]), int(sys.argv[2])
st = State.load(work / "state.json")
st.set("input", "dataset:deep-blending/playroom")
st.start(1, "fetch"); st.finish(1, 0, {"input_bytes": 0})
st.start(2, "frames"); st.finish(2, 0, {"frames": n})
EOF
echo "Dataset prêt : $n images dans $work/2_frames/images"
