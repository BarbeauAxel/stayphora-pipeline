#!/usr/bin/env bash
# Point d'entrée du conteneur sur RunPod :
#  - démarre sshd avec la clé fournie par RunPod ($PUBLIC_KEY) ;
#  - arme un minuteur de sécurité qui SUPPRIME le pod après $MAX_POD_HOURS heures,
#    même si le PC qui a lancé le job a perdu la connexion ;
#  - puis exécute la commande donnée, ou attend (le job est lancé par SSH).
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    # Les variables du pod (secrets R2/Resend) doivent être visibles dans la session SSH.
    : > /etc/stayphora-env.sh
    for v in $(compgen -e | grep -E '^(R2_|RESEND_|RUNPOD_|STAYPHORA_|MAX_POD_HOURS$|AUTO_TERMINATE|PATH$|PYTHONUNBUFFERED$|QT_QPA_PLATFORM$)'); do
        printf 'export %s=%q\n' "$v" "${!v}" >> /etc/stayphora-env.sh
    done
    chmod 600 /etc/stayphora-env.sh
    grep -q stayphora-env /root/.bashrc 2>/dev/null || echo '. /etc/stayphora-env.sh' >> /root/.bashrc
    ssh-keygen -A > /dev/null
    /usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
    echo "sshd démarré"
fi

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    hours="${MAX_POD_HOURS:-3}"
    echo "Minuteur de sécurité : suppression du pod dans ${hours} h"
    ( sleep "$(python3 -c "print(int(float('${hours}') * 3600))")" \
      && echo "Durée maximale atteinte : suppression du pod" \
      && /app/docker/self_terminate.sh ) &
fi

if [[ $# -gt 0 ]]; then
    exec "$@"
fi
exec sleep infinity
