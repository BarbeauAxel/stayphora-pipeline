#!/usr/bin/env bash
# Lance le pipeline en arrière-plan sur le pod (survit à une coupure SSH) :
#   /app/docker/run_job.sh <arguments de « python -m pipeline run »...>
#   ex. /app/docker/run_job.sh --input /workspace/input/video.mp4 --slug studio-lyon
# Journal : /workspace/job.log ; code de sortie : /workspace/job.exit
# Après la fin, si AUTO_TERMINATE_AFTER_MIN est défini, le pod se supprime lui-même après
# ce délai (filet de sécurité si le PC local n'a pas pu le faire).
set -euo pipefail
[[ -f /etc/stayphora-env.sh ]] && . /etc/stayphora-env.sh
rm -f /workspace/job.exit
cd /app
nohup bash -c '
    python -m pipeline run "$@"; code=$?
    echo $code > /workspace/job.exit
    if [[ -n "${AUTO_TERMINATE_AFTER_MIN:-}" ]]; then
        sleep $(( AUTO_TERMINATE_AFTER_MIN * 60 ))
        /app/docker/self_terminate.sh
    fi
' _ "$@" > /workspace/job.log 2>&1 &
echo $! > /workspace/job.pid
echo "Job lancé (pid $(cat /workspace/job.pid))"
