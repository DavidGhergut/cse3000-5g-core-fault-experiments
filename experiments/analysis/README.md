# Cross-modal analysis code

Scripts that produce the results, tables, and figures in the thesis from the
published telemetry dataset.

## Quick start

1. Download the dataset release and extract it so the trial directories live here:

   ```
   experiments/analysis/final_dataset/<trial>/        (boyan, boyan-2, trial4 ... trial8)
   experiments/analysis/final_dataset/security_faults/<attack>/<trial>/
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the whole pipeline:

   ```bash
   bash reproduce.sh
   ```

Generated correlations are written to `correlations_o5g/`, and figures and tables to
`output/`. To run an individual script, run it directly (for example
`python3 significance_tests.py`); each script finds the dataset on its own.

`_paths.py` holds all path configuration. The dataset is expected next to the scripts
by default; set the `DATA_ROOT` environment variable to point elsewhere.

## Pipeline order

`reproduce.sh` runs these stages; the later scripts depend on the artifacts of the
earlier ones.

1. **Correlations.** `cross_correlation.py` (per operational trial) and
   `regen_security_o5g.py` (security faults) compute the cross-modal coupling change
   Δ\|ρ\| for every signal pair and write a `correlations.csv` per trial.
2. **Feature caches.** `op_raw_ablation.py` and `build_feature_cache.py` extract the
   raw per-signal shape-stat features once and cache them under `/tmp`.
3. **Classifiers, ablations, figures.** Everything below.

## Shared modules

| Script | Role |
|--------|------|
| `_paths.py` | Path configuration shared by all scripts |
| `cross_correlation.py` | Operational cross-modal Δ\|ρ\| coupling and lag → `correlations.csv` per trial |
| `regen_security_o5g.py` | Same, for the security faults → `correlations_o5g/security/correlations.csv` |
| `classify_faults_rf.py` | Δ\|ρ\| feature builder + Random Forest + SHAP |
| `classify_extended_features.py` | Raw per-signal shape-stat feature builder (mean/std/maxabs/slope) + classifier |
| `ccf_profile.py` | Full cross-correlation-function profile + Welch coherence |

## Characterization (SQ1–SQ3)

| Script | Output |
|--------|--------|
| `heatmaps_o5g.py` | Per-category Δ\|ρ\| heatmaps (combined and metric sub-layers) |
| `fig_partition_decoupling.py` | Trace–log decoupling under interface partitions (the 7/7 sign-test figure) |
| `significance_tests.py` | Sign test, Wilcoxon, and permutation tests for the partition and metrics–logs findings |

## Classification (SQ4)

| Script | Output |
|--------|--------|
| `compute30_cv.py` | Δ\|ρ\| vs raw vs combined, LOOCV + leave-one-trial-out, 10 seeds (`tab:cv`) |
| `raw_ablation_f1.py` | Raw modality ablation with accuracy and macro-F1 |
| `compute30_delta_abl.py` | Δ\|ρ\| modality ablation (operational and security) |
| `multi_classifier_sq4.py` | Random Forest / logistic regression / gradient boosting robustness |
| `lofo.py` | Leave-one-fault-out (generalization to unseen fault types) |
| `fold_topk_selection.py` | Leakage-free fold-wise top-K feature selection |
| `threshold_sweep.py` | Sparsity-threshold robustness sweep |
| `bin_sweep.py` | Bin-size robustness sweep |
| `learning_curve.py` | Test accuracy vs number of training deployments |
| `raw_intershap.py` | SHAP interaction values (InterSHAP) on the raw model |
| `fig_shap_intershap.py` | SHAP-vs-InterSHAP comparison figure |

## Security faults

| Script | Output |
|--------|--------|
| `security_raw_ablation.py` | Security-fault raw-feature classification (RF and logistic regression) |
| `security_log_volume.py` | Per-attack error-log volume (the security log-blindspot table) |
