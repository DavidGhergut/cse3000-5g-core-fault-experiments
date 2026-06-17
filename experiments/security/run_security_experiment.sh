#!/usr/bin/env bash
# experiments/david/run_security_experiment.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs a security fault scenario using the SAME collection pipeline as the
# C-fault-detection experiments (Prometheus / Loki / Jaeger / RTT / Events).
# The only difference from run_fault.sh is that fault injection uses a custom
# bash script instead of a Chaos Mesh YAML manifest.
#
# Output directory structure matches C-fault-detection exactly:
#   <DATA_DIR>/
#     prometheus/pre|during|post/   ← same CSV files as 22-fault dataset
#     loki/pre|during|post/         ← errors.csv, all.csv, etc.
#     jaeger/pre|during|post/       ← spans_flat.csv
#     rtt/pre|during|post/          ← ue_rtt.csv
#     events/pre|during|post/       ← k8s_events.json
#     nrf/pre|during|post/          ← nrf_registrations.json
#     timeline.json                 ← PRE/FAULT/POST Unix timestamps
#     meta.json
#
# Usage:
#   bash run_security_experiment.sh <fault_type> <trial>
#
# Supported fault types:
#   storm           — NAS registration storm
#   sbi_flood       — HTTP/2 SBI flood on AMF
#   nrf_flood       — HTTP/2 SBI flood on NRF
#   scp_flood       — HTTP/2 SBI flood on SCP (cascades to all inter-NF comms)
#   smf_flood       — HTTP/2 SBI flood on SMF (cascades to UPF + PCF)
#   auth_exhaustion — Authentication exhaustion via AUSF flood
#
# Example:
#   bash run_security_experiment.sh storm 1

set -euo pipefail

FAULT="${1:-storm}"
TRIAL="${2:-1}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"
LIB_DIR="$REPO_ROOT/experiments/lib"

# ── Source shared collection library ─────────────────────────────────────────
# Provides: ensure_portforward_prometheus/jaeger/loki, collect_prometheus,
#           collect_jaeger, collect_loki, collect_loki, collect_events,
#           collect_nrf, now_ts, sleep_with_progress, log_experiment_end, etc.
source "$LIB_DIR/common.sh"
source "$LIB_DIR/traffic.sh"

# ── Output path — mirrors final_dataset/boyan/<fault-name>/ layout ───────────
case "$FAULT" in
    storm)           FAULT_DIRNAME="nas-registration-storm" ;;
    sbi_flood)       FAULT_DIRNAME="sbi-http2-flood-amf" ;;
    nrf_flood)       FAULT_DIRNAME="sbi-http2-flood-nrf" ;;
    scp_flood)       FAULT_DIRNAME="sbi-http2-flood-scp" ;;
    smf_flood)       FAULT_DIRNAME="sbi-http2-flood-smf" ;;
    auth_exhaustion) FAULT_DIRNAME="authentication-exhaustion" ;;
    *)               echo "ERROR: Unknown fault '$FAULT'" >&2; exit 1 ;;
esac

FINAL_DATASET="${DATA_DIR:-$REPO_ROOT/data/5GCore/final_dataset/security_faults}"
OUT_DIR="${FINAL_DATASET}/${FAULT_DIRNAME}/${TRIAL}"

mkdir -p "$OUT_DIR"

# ── Durations (same defaults as C-fault-detection) ───────────────────────────
PRE_DURATION="${PRE_DURATION:-600}"
FAULT_DURATION="${FAULT_DURATION:-300}"
POST_DURATION="${POST_DURATION:-300}"
STEP="${STEP:-5s}"
NAME="${FAULT_DIRNAME}"

export OUT_DIR NAME FAULT_DURATION

echo "════════════════════════════════════════════════════════════════"
echo "  Security Fault: ${FAULT_DIRNAME} | trial=${TRIAL}"
echo "  Output:   ${OUT_DIR}"
echo "  Timeline: ${PRE_DURATION}s pre + ${FAULT_DURATION}s fault + ${POST_DURATION}s post"
echo "  Started:  $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Port-forwards + UE traffic (same as run_fault.sh) ────────────────────────
ensure_portforward_prometheus
ensure_portforward_jaeger
ensure_portforward_loki

_security_cleanup() {
    # Kill any still-running injection process
    [[ -n "${INJECT_PID:-}" ]] && kill "$INJECT_PID" 2>/dev/null || true
    stop_traffic || true
    _cleanup     || true
}
trap _security_cleanup EXIT

start_traffic

# ── collect_phase helper (identical to run_fault.sh) ─────────────────────────
collect_phase() {
    local phase="$1" start="$2" end="$3"
    collect_prometheus "$start" "$end" "$STEP" "$OUT_DIR/prometheus/$phase"
    collect_jaeger     "$start" "$end"         "$OUT_DIR/jaeger/$phase"
    collect_loki       "$start" "$end"         "$OUT_DIR/loki/$phase"
    collect_events     "$start" "$end"         "$OUT_DIR/events/$phase"
    collect_nrf                                "$OUT_DIR/nrf/$phase"
}

# ── PRE window ───────────────────────────────────────────────────────────────
echo "[fault] PRE window (${PRE_DURATION}s)..."
PRE_START=$(now_ts)
mkdir -p "$OUT_DIR/rtt/pre"
bash "$LIB_DIR/collect_ue_rtt.sh" "$PRE_DURATION" "$OUT_DIR/rtt/pre/ue_rtt.csv" &
PRE_RTT_PID=$!
sleep_with_progress "$PRE_DURATION" "pre-fault baseline"
PRE_END=$(now_ts)
wait "$PRE_RTT_PID" 2>/dev/null || true
collect_phase pre "$PRE_START" "$PRE_END"

# ── Orphaned-bearer guard (same check as run_fault.sh) ───────────────────────
PRE_ERR="$OUT_DIR/loki/pre/errors.csv"
if [[ -f "$PRE_ERR" ]] && ! python3 - "$PRE_ERR" <<'PYEOF'
import csv, re, sys, collections
rows = list(csv.reader(open(sys.argv[1], newline='')))
body = rows[1:] if rows else []
if not body:
    sys.exit(0)
teid = collections.Counter()
for r in body:
    line = r[-1] if r else ""
    if "Send Error Indication" in line:
        m = re.search(r"TEID:0x[0-9a-fA-F]+", line)
        teid[m.group(0) if m else "?"] += 1
top = max(teid.values()) if teid else 0
sys.exit(1 if top > 0.30 * len(body) else 0)
PYEOF
then
    echo "FATAL: PRE baseline contaminated by an orphaned-bearer SEI flood" \
         "(single TEID > 30% of $PRE_ERR). Discard & re-run." >&2
    exit 1
fi

# ── Inject security fault ─────────────────────────────────────────────────────
echo "[fault] Injecting fault: ${FAULT_DIRNAME} (${FAULT_DURATION}s)..."
FAULT_START=$(now_ts)

case "$FAULT" in
    storm)
        bash "$SCRIPTS_DIR/inject_registration_storm.sh" 40 "$FAULT_DURATION" &
        ;;
    sbi_flood)
        bash "$SCRIPTS_DIR/inject_sbi_flood.sh" 20 "$FAULT_DURATION" &
        ;;
    nrf_flood)
        bash "$SCRIPTS_DIR/inject_nrf_sbi_flood.sh" 20 "$FAULT_DURATION" &
        ;;
    scp_flood)
        bash "$SCRIPTS_DIR/inject_sbi_flood_scp.sh" 20 "$FAULT_DURATION" &
        ;;
    smf_flood)
        bash "$SCRIPTS_DIR/inject_sbi_flood_smf.sh" 20 "$FAULT_DURATION" &
        ;;
    auth_exhaustion)
        bash "$SCRIPTS_DIR/inject_auth_exhaustion.sh" 20 "$FAULT_DURATION" &
        ;;
esac
INJECT_PID=$!

# ── DURING window ─────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR/rtt/during"
bash "$LIB_DIR/collect_ue_rtt.sh" "$FAULT_DURATION" "$OUT_DIR/rtt/during/ue_rtt.csv" &
UE_RTT_PID=$!
sleep_with_progress "$FAULT_DURATION" "fault active"
FAULT_END=$(now_ts)
wait "$UE_RTT_PID" 2>/dev/null || true

# Ensure injection process has stopped
wait "$INJECT_PID" 2>/dev/null || true
INJECT_PID=""

collect_phase during "$FAULT_START" "$FAULT_END"

# ── POST window ───────────────────────────────────────────────────────────────
echo "[fault] POST window (${POST_DURATION}s)..."
REMOVE_TS=$(now_ts)
mkdir -p "$OUT_DIR/rtt/post"
bash "$LIB_DIR/collect_ue_rtt.sh" "$POST_DURATION" "$OUT_DIR/rtt/post/ue_rtt.csv" &
POST_RTT_PID=$!
sleep_with_progress "$POST_DURATION" "post-fault recovery"
POST_END=$(now_ts)
wait "$POST_RTT_PID" 2>/dev/null || true
collect_phase post "$REMOVE_TS" "$POST_END"

# ── Write timeline.json (identical format to run_fault.sh) ───────────────────
python3 -c "
import json
timeline = {
    'name':  '${NAME}',
    'pre':   {'start': ${PRE_START},   'end': ${PRE_END}},
    'fault': {'start': ${FAULT_START}, 'end': ${FAULT_END}},
    'post':  {'start': ${REMOVE_TS},   'end': ${POST_END}},
}
with open('${OUT_DIR}/timeline.json', 'w') as f:
    json.dump(timeline, f, indent=2)
print('[timeline] written')
"

log_experiment_end "$OUT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ ${FAULT_DIRNAME} trial ${TRIAL} complete"
echo "  Output: ${OUT_DIR}"
echo "  Structure mirrors C-fault-detection dataset exactly."
echo "════════════════════════════════════════════════════════════════"
