"""Bin-size robustness sweep.

Re-runs the correlation pipeline at several bin sizes (5-30 s) by re-executing
cross_correlation.py with patched BIN_SEC / N_BINS, and reports, for each bin
size, the resulting feature count, LOOCV accuracy, and the per-category
metrics-logs Delta|rho| heatmap. Shows the findings are stable across bin size.
"""

import re
import types
import warnings
from pathlib import Path

import _paths
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score

CC_SOURCE = open(_paths.HERE / "cross_correlation.py").read()
TRIAL_DIRS = {t: str(_paths.trial_dir(t)) for t in _paths.TRIALS}
CROSS_PAIRS = ["metrics↔logs", "metrics↔traces", "traces↔logs"]
CATEGORIES = ["infrastructure", "network", "pfcp", "pod_crash"]


def features_and_accuracy(corr):
    """Return (per-category metrics-logs heatmap, feature-matrix shape, mean LOOCV accuracy)."""
    corr = corr[corr.modality_pair.isin(CROSS_PAIRS)].copy()
    corr["r"] = pd.to_numeric(corr.spearman_r, errors="coerce")
    corr["cat"] = corr.fault_category.replace({"cascade": "pod_crash"})

    # Delta|rho| per signal pair.
    delta_table = corr.pivot_table(
        index=["fault_name", "trial", "cat", "sig_a", "sig_b", "modality_pair"],
        columns="window", values="r").reset_index()
    delta_table = delta_table.dropna(subset=["pre", "during"])
    delta_table["d"] = delta_table.during.abs() - delta_table.pre.abs()

    # Mean metrics-logs Delta|rho| per category (heatmap row).
    heatmap = {c: delta_table[(delta_table.cat == c) & (delta_table.modality_pair == "metrics↔logs")]["d"].mean()
               for c in CATEGORIES}

    # Per-instance feature matrix for the classifier.
    instance_matrix = delta_table.pivot_table(index=["fault_name", "trial", "cat"],
                                              columns=["sig_a", "sig_b"], values="d", fill_value=0.0)
    y = instance_matrix.index.get_level_values("cat").values
    X = instance_matrix.values
    keep_cols = (X != 0).mean(0) >= 0.20
    X = X[:, keep_cols]
    accuracies = [accuracy_score(y, cross_val_predict(
        RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=seed),
        X, y, cv=LeaveOneOut(), n_jobs=-1)) for seed in range(5)]
    return heatmap, X.shape, np.mean(accuracies)


print(f"{'bin':>5}{'bins':>6}{'feats':>8}{'LOOCV acc':>11}   "
      f"heatmap metrics-logs Δ|ρ| (infra/net/pfcp/pod)", flush=True)
for bin_sec in [5, 10, 15, 20, 30]:
    n_bins = 300 // bin_sec
    # Re-execute cross_correlation.py with patched bin parameters.
    patched_source = re.sub(r"BIN_SEC\s*=\s*10", f"BIN_SEC   = {bin_sec}", CC_SOURCE)
    patched_source = re.sub(r"N_BINS\s*=\s*30", f"N_BINS    = {n_bins}", patched_source)
    cc = types.ModuleType("cc")
    cc.__dict__["__name__"] = "cc"
    exec(compile(patched_source, "cc", "exec"), cc.__dict__)

    rows = []
    for trial, trial_dir in TRIAL_DIRS.items():
        for fault_dir in sorted(Path(trial_dir).iterdir()):
            if fault_dir.is_dir() and fault_dir.name in cc.FAULT_CATEGORIES:
                for row in cc.analyze_fault(fault_dir):
                    row["fault_name"] = fault_dir.name
                    row["fault_category"] = cc.FAULT_CATEGORIES[fault_dir.name]
                    row["trial"] = trial
                    rows.append(row)
    heatmap, shape, acc = features_and_accuracy(pd.DataFrame(rows))
    heatmap_str = "  ".join(f"{heatmap[c]:+.3f}" for c in CATEGORIES)
    print(f"{bin_sec:>4}s{n_bins:>6}{shape[1]:>8}{acc * 100:>10.1f}%     {heatmap_str}", flush=True)
print("DONE")
