#!/usr/bin/env bash
# Supprime (terminate) le pod courant via l'API RunPod.
# Clé utilisée : STAYPHORA_TERMINATE_KEY si fournie, sinon RUNPOD_API_KEY (injectée par RunPod
# dans chaque pod, « pod-scoped »).
set -uo pipefail
[[ -f /etc/stayphora-env.sh ]] && . /etc/stayphora-env.sh
pod="${RUNPOD_POD_ID:?RUNPOD_POD_ID absent : pas sur RunPod ?}"
key="${STAYPHORA_TERMINATE_KEY:-${RUNPOD_API_KEY:-}}"
if [[ -z "$key" ]]; then
    echo "ERREUR : aucune clé API pour supprimer le pod $pod" >&2
    exit 1
fi
for attempt in 1 2 3 4 5; do
    if RUNPOD_API_KEY="$key" /opt/venv/bin/python - "$pod" <<'EOF'
import os, sys
import runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(sys.argv[1])
EOF
    then
        echo "Pod $pod supprimé"
        exit 0
    fi
    echo "Échec de la suppression (tentative $attempt), nouvel essai dans 30 s" >&2
    sleep 30
done
exit 1
