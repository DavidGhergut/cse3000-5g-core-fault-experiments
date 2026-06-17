#!/usr/bin/env bash
# run_auth_exhaustion_trials.sh
# ─────────────────────────────
# Runs 3 independent trials of the Authentication Exhaustion security fault.
# Each trial: 600s baseline + 300s fault + 300s recovery (~20 min each, ~1h total).
#
# Output lands in:
#   final_dataset/security_faults/authentication-exhaustion/1/
#   final_dataset/security_faults/authentication-exhaustion/2/
#   final_dataset/security_faults/authentication-exhaustion/3/
#
# Prerequisites:
#   - kind cluster 'open5gs' must be running and healthy
#   - kubectl context set to kind-open5gs
#   - Port-forwards will be started automatically
#
# Usage:
#   bash run_auth_exhaustion_trials.sh
#   bash run_auth_exhaustion_trials.sh --only 2   # run just trial 2

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "════════════════════════════════════════════════════════"
echo "  Authentication Exhaustion — 3 trial run"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════"

for TRIAL in 1 2 3; do
    if [[ -n "$ONLY" && "$TRIAL" != "$ONLY" ]]; then
        echo "[skip] Trial $TRIAL"
        continue
    fi

    OUT="$SCRIPTS_DIR/../../data/5GCore/final_dataset/security_faults/authentication-exhaustion/${TRIAL}"
    if [[ -d "$OUT" && -f "$OUT/timeline.json" ]]; then
        echo "[skip] Trial $TRIAL already exists at $OUT"
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  Trial $TRIAL / 3"
    echo "────────────────────────────────────────────────────"
    bash "$SCRIPTS_DIR/run_security_experiment.sh" auth_exhaustion "$TRIAL"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "✓ All trials complete"
echo "  Finished: $(date)"
echo ""
echo "Next: run cross_correlation.py on security_faults to include"
echo "      authentication-exhaustion in the analysis."
echo "════════════════════════════════════════════════════════"
