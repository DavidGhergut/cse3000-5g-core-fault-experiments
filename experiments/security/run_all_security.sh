#!/usr/bin/env bash
# experiments/david/run_all_security.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs all security fault scenarios for N trials each, using the same
# collection pipeline as the C-fault-detection experiments.
#
# Output lands in final_dataset/security_faults/<fault-name>/<trial>/
# with the identical directory structure to final_dataset/boyan/<fault-name>/
#
# Usage:
#   bash run_all_security.sh              # 3 trials each, all faults
#   bash run_all_security.sh --trials 1   # 1 trial each
#   bash run_all_security.sh --from storm # resume from a specific fault

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIALS=3
FROM_FAULT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials) TRIALS="$2"; shift 2 ;;
        --from)   FROM_FAULT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

FAULTS=(storm sbi_flood nrf_flood scp_flood smf_flood auth_exhaustion)
FAULT_NAMES=(nas-registration-storm sbi-http2-flood-amf sbi-http2-flood-nrf sbi-http2-flood-scp sbi-http2-flood-smf authentication-exhaustion)

echo "════════════════════════════════════════════════════════════════"
echo "  Security Fault Dataset Collection"
echo "  Faults: ${#FAULTS[@]} scenarios × ${TRIALS} trials each"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

SKIP=false
[[ -n "$FROM_FAULT" ]] && SKIP=true

for i in "${!FAULTS[@]}"; do
    FAULT="${FAULTS[$i]}"
    DIRNAME="${FAULT_NAMES[$i]}"

    # --from support
    if $SKIP; then
        [[ "$FAULT" == "$FROM_FAULT" ]] && SKIP=false || { echo "[skip] $DIRNAME"; continue; }
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Fault: ${DIRNAME}"
    echo "════════════════════════════════════════════════════════════════"

    for TRIAL in $(seq 1 "$TRIALS"); do
        REPO_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"
        OUT="${DATA_DIR:-$REPO_ROOT/data/5GCore/final_dataset/security_faults}/${DIRNAME}/${TRIAL}"

        if [[ -d "$OUT" && -f "$OUT/timeline.json" ]]; then
            echo "[skip] ${DIRNAME} trial ${TRIAL} — already complete"
            continue
        fi

        echo ""
        echo "  ── Trial ${TRIAL} / ${TRIALS} ──"
        bash "$SCRIPTS_DIR/run_security_experiment.sh" "$FAULT" "$TRIAL"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ All security fault trials complete"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════════"
