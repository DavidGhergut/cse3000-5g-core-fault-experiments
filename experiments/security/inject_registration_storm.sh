#!/usr/bin/env bash
# Inject a registration storm on AMF by rapidly spawning UERANSIM UE registrations.
# This simulates a NAS flooding / signaling storm security attack.
#
# Usage: ./inject_registration_storm.sh [num_ues] [duration_seconds]
# Default: 40 UEs, 300 seconds
#
# Before running:
#   1. Verify UE pod name: kubectl get pods --all-namespaces | grep ue
#   2. Make sure collect.py / run_security_experiment.sh is already running

set -euo pipefail

NUM_UES=${1:-40}
DURATION=${2:-300}

echo "[storm] Finding UERANSIM UE pod..."

# Auto-detect UE pod and namespace
UE_NS=$(kubectl get pods --all-namespaces | grep -i "ue" | grep -v "true\|ueransim-gnb\|Terminating" | head -1 | awk '{print $1}')
UE_POD=$(kubectl get pods --all-namespaces | grep -i "ue" | grep -v "true\|ueransim-gnb\|Terminating" | head -1 | awk '{print $2}')

if [[ -z "$UE_POD" ]]; then
    echo "ERROR: Could not find UE pod. Run: kubectl get pods --all-namespaces | grep ue"
    exit 1
fi

echo "[storm] UE pod: $UE_POD (namespace: $UE_NS)"

# Resolve gNB IP and create a patched config with the current IP.
# /etc/ueransim/ue.yaml has ${GNB_IP} as a placeholder that envsubst must fill.
# The pod's /ue.yaml was created at start-up and may be stale if the gNB pod
# was restarted; always re-generate from /etc/ueransim/ue.yaml.
GNB_IP=$(kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c \
    'getent hosts $GNB_HOSTNAME | head -1 | cut -d" " -f1' 2>/dev/null || echo "")
if [[ -z "$GNB_IP" ]]; then
    echo "ERROR: Could not resolve gNB IP from GNB_HOSTNAME inside pod"
    exit 1
fi

kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c \
    "GNB_IP=${GNB_IP}; export GNB_IP; envsubst < /etc/ueransim/ue.yaml > /tmp/ue_storm.yaml"
UE_CONFIG="/tmp/ue_storm.yaml"

echo "[storm] UE config: $UE_CONFIG (gNB IP: $GNB_IP)"
echo "[storm] Storm duration: ${DURATION}s in waves of ${NUM_UES} UEs"
echo "[storm] Each wave triggers: UE → gNB → AMF → AUSF → UDM auth chain"
echo "[storm] Note: only IMSI 1 is in subscriber DB — rest fail auth but still flood AMF"
echo ""

# Loop waves of registrations for the full fault duration
START_TIME=$(date +%s)
END_TIME=$(( START_TIME + DURATION ))
WAVE=1

while [[ $(date +%s) -lt $END_TIME ]]; do
    REMAINING=$(( END_TIME - $(date +%s) ))
    WAVE_TIMEOUT=$(( REMAINING < 30 ? REMAINING : 30 ))

    echo "[storm] Wave $WAVE — ${REMAINING}s remaining, timeout=${WAVE_TIMEOUT}s"

    kubectl exec -n "$UE_NS" "$UE_POD" -- \
        timeout "$WAVE_TIMEOUT" nr-ue -c "$UE_CONFIG" -n "$NUM_UES" 2>&1 | tail -3 || true

    WAVE=$(( WAVE + 1 ))
    sleep 1
done

echo ""
echo "[storm] Done. Ran $WAVE waves over ${DURATION}s."
echo "[storm] Telemetry expected:"
echo "  - Prometheus: AMF cpu_rate spike"
echo "  - Loki: flood of Registration Request / auth rejection messages on AMF"
echo "  - Jaeger: AMF→AUSF and AMF→NRF call count explosion"
