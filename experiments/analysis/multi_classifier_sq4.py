"""Multi-classifier robustness for SQ4.

Runs Random Forest, logistic regression, and gradient boosting on the raw
shape-stat features (multi-modal vs metrics-only) and on the Delta|rho| coupling
features, under LOOCV and leave-one-trial-out. Shows that across classifiers
(1) raw features beat Delta|rho|, and (2) multi-modal is close to metrics-only.
"""

import re

import _paths
import numpy as np
import pandas as pd
import classify_extended_features as ext
import classify_faults_rf as cf
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

SEED = 42


def trial_of(fault_id):
    match = re.search(r"\[([^\]]+)\]", fault_id)
    return match.group(1) if match else fault_id.split()[-1]


def classifiers():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                               class_weight="balanced", random_state=SEED, n_jobs=-1),
        "LogisticReg": make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=3000, class_weight="balanced")),
        "GradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def evaluate(X, y, groups, name):
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    print(f"\n  [{name}]  ({X.shape[0]} inst, {X.shape[1]} feat)")
    print(f"    {'classifier':<18}{'LOOCV':>9}{'LOTO':>9}")
    for clf_name, clf in classifiers().items():
        loocv = (cross_val_predict(clf, X, y, cv=LeaveOneOut()) == y).mean()
        loto = (cross_val_predict(clf, X, y, cv=LeaveOneGroupOut(), groups=groups) == y).mean()
        print(f"    {clf_name:<18}{loocv * 100:>7.1f}%{loto * 100:>8.1f}%")


def feature_modality(col):
    modality = ext.modality_of(col)
    return "metrics" if modality == "rtt" else modality


# Raw shape-stat features.
roots = [_paths.trial_dir(t) for t in _paths.TRIALS]
print("loading raw shape-stat features ...", flush=True)
raw = pd.concat([ext.load_dataset([r], ext.OPERATIONAL_CATEGORIES, ["mean", "std", "maxabs", "slope"])
                 for r in roots], ignore_index=True)
X_raw, y_raw, ids_raw, features_raw = ext.build_matrix(raw, ext.CATS_OPERATIONAL, ext.RELABEL)
groups_raw = np.array([trial_of(i) for i in ids_raw])
metric_cols = [i for i, col in enumerate(features_raw) if feature_modality(col) == "metrics"]

print("=" * 48)
print("RAW SHAPE-STAT FEATURES")
print("=" * 48)
evaluate(X_raw, y_raw, groups_raw, "raw — ALL modalities (multi-modal)")
evaluate(np.asarray(X_raw)[:, metric_cols], y_raw, groups_raw, "raw — metrics only (single-modality)")

# Delta|rho| cross-modal coupling features.
corr_paths = [str(_paths.corr_csv(t)) for t in _paths.TRIALS]
cfg = cf.MODE_CONFIGS["operational"]
X_delta, y_delta, ids_delta, _ = cf.build_feature_matrix(
    corr_paths, cfg["cats"], cfg["relabel"], cf.MODAL_SETS["cross"])
groups_delta = np.array([trial_of(i) for i in ids_delta])

print("\n" + "=" * 48)
print("Δ|ρ| CROSS-MODAL COUPLING FEATURES")
print("=" * 48)
evaluate(X_delta, y_delta, groups_delta, "Δ|ρ| — all cross-modal")

print("\nseed=42, single run; HistGradientBoosting used for 'gradient boosting'.")
print("[done]")
