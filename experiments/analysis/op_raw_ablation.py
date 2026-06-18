"""Operational RAW modality ablation for the main-body SQ4 table.

Trains a Random Forest on the raw per-signal shape-stat features, restricted to
each modality subset (singles, pairs, and all three), and reports leave-one-out
(LOOCV) and leave-one-trial-out (LOTO) accuracy over the 7 operational trials.

The feature matrix is cached to /tmp so re-runs are fast.
"""

import os
import re

import _paths
import numpy as np
import pandas as pd
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

CACHE = "/tmp/op_raw_ablation.npz"


def trial_of(fault_id):
    """Extract the trial name from a fault id of the form 'fault [trial]'."""
    match = re.search(r"\[([^\]]+)\]", fault_id)
    return match.group(1) if match else fault_id


# Build the raw feature matrix once (X = features, y = category, groups = trial).
if os.path.exists(CACHE):
    cached = np.load(CACHE, allow_pickle=True)
    X, y, groups, feature_names = cached["X"], cached["y"], cached["g"], list(cached["feat"])
    print("cache", flush=True)
else:
    frames = []
    for trial in _paths.TRIALS:
        print(f"  loading {trial}", flush=True)
        df = ext.load_dataset([_paths.trial_dir(trial)], ext.OPERATIONAL_CATEGORIES,
                              ["mean", "std", "maxabs", "slope"])
        df["dataset"] = trial
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X, y, ids, feature_names = ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
    X, y = np.asarray(X), np.asarray(y)
    groups = np.array([trial_of(fault_id) for fault_id in ids])
    np.savez(CACHE, X=X, y=y, g=groups, feat=np.array(feature_names, dtype=object))

print("trials:", pd.Series(groups).value_counts().to_dict(),
      "n=", len(y), "feat=", len(feature_names), flush=True)


def feature_modality(col):
    """Modality of a feature column; RTT is grouped with metrics."""
    modality = ext.modality_of(col)
    return "metrics" if modality == "rtt" else modality


feature_mods = np.array([feature_modality(col) for col in feature_names])


def columns_for(subset):
    """Indices of the feature columns belonging to a '+'-separated modality subset."""
    wanted = set(subset.split("+"))
    return [i for i, modality in enumerate(feature_mods) if modality in wanted]


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


majority_baseline = max(pd.Series(y).value_counts()) / len(y)
print(f"majority-class baseline {majority_baseline * 100:.1f}%\n")
print(f"{'subset':16}{'#feat':>6}{'LOOCV':>8}{'LOTO':>8}")

SUBSETS = ["metrics", "logs", "traces", "metrics+logs", "metrics+traces",
           "logs+traces", "metrics+logs+traces"]
for subset in SUBSETS:
    col_idx = columns_for(subset)
    X_subset = X[:, col_idx]
    loocv = (cross_val_predict(make_rf(), X_subset, y, cv=LeaveOneOut()) == y).mean()
    loto = (cross_val_predict(make_rf(), X_subset, y, cv=LeaveOneGroupOut(), groups=groups) == y).mean()
    label = "ALL" if subset == "metrics+logs+traces" else subset
    print(f"{label:16}{len(col_idx):>6}{loocv * 100:>7.1f}{loto * 100:>7.1f}")
print("[done]")
