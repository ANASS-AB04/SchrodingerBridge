#!/usr/bin/env bash
# Vérifie qu'une copie locale de la sauvegarde correspond à celle du cluster.
#
#     ./verifier-copie.sh ~/backup-stage
#
# Compare nombre de fichiers, volume total et empreintes d'un échantillon, des
# deux côtés. À lancer après le rsync de rapatriement, avant de considérer que
# la copie est bonne.
set -uo pipefail

LOCAL="${1:-}"
REMOTE_HOST="${REMOTE_HOST:-aaboufadel@cali3.unilim.fr}"
REMOTE_DIR="${REMOTE_DIR:-/scratch/aaboufadel/backup-stage}"

[[ -z "$LOCAL" ]] && { echo "usage: $0 <dossier-local>"; exit 1; }
[[ -d "$LOCAL" ]] || { echo "introuvable : $LOCAL"; exit 1; }

# Les compteurs attendus, mesurés et vérifiés le 18/09/2026.
EXPECT_FIELDS=1433
EXPECT_PNG=89994

echo "local   : $LOCAL"
echo "distant : $REMOTE_HOST:$REMOTE_DIR"
echo

stats() {  # $1 = racine ; imprime "fichiers octets fields png"
    local r="$1"
    echo "$(find "$r" -type f | wc -l) \
$(find "$r" -type f -printf '%s\n' | awk '{t+=$1} END{print t+0}') \
$(find "$r" -name fields.npz | wc -l) \
$(find "$r" -name '*.png' | wc -l)"
}

read -r L_N L_B L_F L_P <<< "$(stats "$LOCAL")"
read -r R_N R_B R_F R_P <<< "$(ssh -o BatchMode=yes "$REMOTE_HOST" \
    "$(declare -f stats); stats '$REMOTE_DIR'" 2>/dev/null | tail -1)"

printf '%-14s %12s %16s %10s %10s\n' "" fichiers octets fields.npz png
printf '%-14s %12s %16s %10s %10s\n' local   "$L_N" "$L_B" "$L_F" "$L_P"
printf '%-14s %12s %16s %10s %10s\n' cluster "$R_N" "$R_B" "$R_F" "$R_P"
echo

fail=0
[[ "$L_N" == "$R_N" ]] || { echo "ECHEC : nombre de fichiers différent"; fail=1; }
[[ "$L_B" == "$R_B" ]] || { echo "ECHEC : volume différent"; fail=1; }
[[ "$L_F" == "$EXPECT_FIELDS" ]] || { echo "ECHEC : $L_F fields.npz au lieu de $EXPECT_FIELDS"; fail=1; }
[[ "$L_P" == "$EXPECT_PNG" ]] || { echo "ECHEC : $L_P png au lieu de $EXPECT_PNG"; fail=1; }

# Échantillon d'empreintes : trois fields.npz pris au début, au milieu et à la fin.
echo "empreintes sur échantillon…"
SAMPLE=$(cd "$LOCAL" && find . -name fields.npz | sort | sed -n '1p;700p;1433p')
for f in $SAMPLE; do
    a=$(cd "$LOCAL" && md5sum "$f" | cut -d' ' -f1)
    b=$(ssh -o BatchMode=yes "$REMOTE_HOST" "cd '$REMOTE_DIR' && md5sum '$f'" \
        2>/dev/null | cut -d' ' -f1)
    if [[ "$a" == "$b" && -n "$a" ]]; then
        echo "  OK    ${f#./}"
    else
        echo "  ECHEC ${f#./}  ($a / $b)"; fail=1
    fi
done

echo
if [[ "$fail" == 0 ]]; then
    echo "COPIE CONFORME — le cluster peut être libéré."
else
    echo "DIFFERENCES DETECTEES — relancer le rsync, ne rien supprimer."
    exit 1
fi
