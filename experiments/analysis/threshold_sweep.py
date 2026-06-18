"""Sparsity-threshold robustness sweep for both feature models.

Sweeps the minimum-presence filter (10-30%) for the Delta|rho| and raw feature
matrices and reports 5-seed LOOCV and leave-one-trial-out accuracy, showing the
classification accuracy is stable around the 20% threshold the paper uses.
"""

import re

import _paths
import numpy as np
import pandas as pd
import classify_faults_rf as cf
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


def trial_of(fault_id):
    """Extract the trial name from a fault id of the form 'fault [trial]'."""
    match = re.search(r"\[([^\]]+)\]", fault_id)
    return match.group(1) if match else fault_id


def make_rf(seed):
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def mean_acc_5seeds(X, y, groups):
    """Mean LOOCV and LOTO accuracy (%) over 5 seeds."""
    loocv, loto = [], []
    for seed in range(5):
        loocv.append((cross_val_predict(make_rf(seed), X, y, cv=LeaveOneOut()) == y).mean())
        loto.append((cross_val_predict(make_rf(seed), X, y, cv=LeaveOneGroupOut(), groups=groups) == y).mean())
    return np.mean(loocv) * 100, np.mean(loto) * 100


# Delta|rho| coupling features.
print("=== DELTA|rho| sparsity-threshold sweep (5 seeds) ===", flush=True)
corr_paths = [str(_paths.corr_csv(t)) for t in _paths.TRIALS]
cfg = cf.MODE_CONFIGS["operational"]
for min_frac in THRESHOLDS:
    X, y, ids, _ = cf.build_feature_matrix(corr_paths, cfg["cats"], cfg["relabel"], None, min_frac=min_frac)
    groups = np.array([trial_of(i) for i in ids])
    loocv, loto = mean_acc_5seeds(X, y, groups)
    print(f"  thr={min_frac:.2f}  {X.shape[1]:4d}f  LOOCV {loocv:5.1f}  LOTO {loto:5.1f}", flush=True)

# Raw shape-stat features (from the per-trial feature cache).
print("=== RAW sparsity-threshold sweep (5 seeds) ===", flush=True)
features = pd.concat([pd.read_pickle(f"/tmp/featcache/{t}.pkl") for t in _paths.TRIALS], ignore_index=True)
for min_frac in THRESHOLDS:
    X, y, ids, _ = ext.build_matrix(features, ext.CATS_OPERATIONAL, ext.RELABEL, min_frac=min_frac)
    X, y = np.asarray(X), np.asarray(y)
    groups = np.array([trial_of(i) for i in ids])
    loocv, loto = mean_acc_5seeds(X, y, groups)
    print(f"  thr={min_frac:.2f}  {X.shape[1]:4d}f  LOOCV {loocv:5.1f}  LOTO {loto:5.1f}", flush=True)
print("[done]", flush=True)
