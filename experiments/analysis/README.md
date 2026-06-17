# Paper analysis code

Scripts that produce the results, tables, and figures in the thesis. They read the
per-fault experiment data (`data/5GCore/final_dataset/<trial>/`) and the
cross-correlation outputs (`data/5GCore/correlations/<trial>/`), and the figure
scripts write PNGs to the paper's `figures/` directory.

## Core modules (imported by the others)

| Script | Role |
|--------|------|
| `cross_correlation.py` | Cross-modal Δ\|ρ\| coupling + lag-zero Spearman and rank cross-correlation lag → `correlations.csv` per trial |
| `classify_extended_features.py` | Raw per-signal shape-stat feature builder (mean/std/maxabs/slope) |
| `classify_faults_rf.py` | Δ\|ρ\| feature builder + Random Forest classifier + SHAP |

## Characterization (SQ1–SQ3)

| Script | Output |
|--------|--------|
| `regen_delta_heatmap.py` | Per-category Δ\|ρ\| heatmap figure |
| `fig_partition_decoupling.py` | Trace–log decoupling under interface partitions figure |
| `ccf_profile.py`, `ccf_lag_summary.py` | Full rank cross-correlation profiles + non-zero-peak-lag statistic |
| `validate_sq3_findings.py`, `validate_sq3_stage2.py` | Validation of the SQ3 coupling findings |
| `significance_tests.py` | Significance tests for the partition decoupling and metrics–logs positivity |

## Classification (SQ4)

| Script | Output |
|--------|--------|
| `op_raw_ablation.py` | Raw-feature modality ablation table |
| `modality_ablation_perclass.py` | Per-category modality ablation |
| `classify_trial_cv.py` | Leave-one-instance-out vs leave-one-trial-out comparison |
| `combined_loto2.py`, `combined_seeds.py` | Raw + Δ\|ρ\| combined model (point estimate and 30-seed) |
| `multi_classifier_sq4.py` | RF / logistic-regression / gradient-boosting robustness |
| `fold_topk_selection.py` | Leakage-free fold-wise top-K feature selection |
| `pca_sanity.py` | PCA→RF comparison |
| `lofo.py` | Leave-one-fault-out (generalization to unseen fault types) |
| `upf_robustness.py` | Ablation restricted to the UPF-equipped trials |
| `raw_intershap.py` | SHAP interaction values (InterSHAP) on the raw model |
| `fig_shap_intershap.py` | SHAP-vs-InterSHAP comparison figure |
| `shap_modality_fair.py`, `shap_raw_perclass.py` | SHAP attribution figures |
| `learning_curve.py` | Data-sufficiency learning curve figure |

## Security faults

| Script | Output |
|--------|--------|
| `cross_correlation_security.py` | Cross-modal Δ\|ρ\| for the security faults |
| `security_raw_ablation.py` | Security-fault raw-feature classification |

## Figure

| Script | Output |
|--------|--------|
| `fig_pipeline.py` | Testbed + cross-modal processing pipeline diagram |

## Notes

- Figure scripts write to `/Users/david/Desktop/Research paper_david/figures/`. Adjust
  the `FIG` / output path constant at the top of each script when running elsewhere.
- Data paths are relative to the project root (`data/5GCore/...`); run from there with
  the analysis dependencies installed (numpy, pandas, scipy, scikit-learn, matplotlib,
  shap).
