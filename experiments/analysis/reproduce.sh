#!/usr/bin/env bash
#
# Reproduce the cross-modal analysis figures and tables from the dataset.
#
# 1. Extract the dataset release so that the trial directories live under
#       experiments/analysis/final_dataset/<trial>/
# 2. pip install -r requirements.txt
# 3. bash reproduce.sh
#
# Generated correlations land in correlations_o5g/, figures and tables in output/.
#
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

TRIALS=(boyan boyan-2 trial4 trial5 trial6 trial7 trial8)

trial_path() {  # echo the fault directory for a trial (handles the C-fault-detection subdir)
    local t="final_dataset/$1"
    [ -d "$t/C-fault-detection" ] && t="$t/C-fault-detection"
    echo "$t"
}

echo "== 1/5 operational cross-modal correlations (per trial) =="
for t in "${TRIALS[@]}"; do
    "$PY" cross_correlation.py --data "$(trial_path "$t")" --out "correlations_o5g/$t"
done

echo "== 2/5 security cross-modal correlations =="
"$PY" regen_security_o5g.py

echo "== 3/5 feature caches (raw ablation + per-trial feature cache) =="
"$PY" op_raw_ablation.py
"$PY" build_feature_cache.py

echo "== 4/5 classifiers, ablations, robustness =="
DATA_ARGS=()
for t in "${TRIALS[@]}"; do DATA_ARGS+=("$(trial_path "$t")"); done
"$PY" classify_extended_features.py --data "${DATA_ARGS[@]}" --out output/classifier_extended
CORR_ARGS=()
for t in "${TRIALS[@]}"; do CORR_ARGS+=("correlations_o5g/$t/correlations.csv"); done
"$PY" classify_faults_rf.py --corr "${CORR_ARGS[@]}" --out output/classifier_delta
"$PY" compute30_cv.py
"$PY" raw_ablation_f1.py
"$PY" compute30_delta_abl.py
"$PY" multi_classifier_sq4.py
"$PY" lofo.py
"$PY" threshold_sweep.py
"$PY" fold_topk_selection.py
"$PY" learning_curve.py
"$PY" security_raw_ablation.py

echo "== 5/5 figures, significance tests, security log blindspot =="
"$PY" heatmaps_o5g.py
"$PY" fig_partition_decoupling.py
"$PY" significance_tests.py
"$PY" security_log_volume.py
"$PY" fig_shap_intershap.py

echo "Done. Correlations in correlations_o5g/, figures and tables in output/."
