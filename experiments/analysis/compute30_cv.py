"""tab:cv: Delta|rho|, raw, and combined features, LOOCV + leave-one-trial-out.

Reports mean +/- std accuracy and macro-F1 over multiple seeds for the three
feature views. The raw and per-trial feature matrices come from caches built by
op_raw_ablation.py and build_feature_cache.py.
"""

import re

import _paths  # noqa: F401  (anchors sys.path and dataset paths)
import numpy as np
import pandas as pd
import classify_faults_rf as cf
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import f1_score

SEEDS = 10
OPERATIONAL_CATS = ["infrastructure", "network", "pfcp", "pod_crash"]


def trial_of(fault_id):
    """Extract the trial name from a fault id of the form 'fault [trial]'."""
    return re.search(r"\[([^\]]+)\]", fault_id).group(1)


def make_rf(seed):
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def report(name, X, y, groups):
    """Print mean +/- std accuracy and macro-F1 for LOOCV and LOTO over SEEDS seeds."""
    loocv_acc, loocv_f1, loto_acc, loto_f1 = [], [], [], []
    for seed in range(SEEDS):
        pred_loocv = cross_val_predict(make_rf(seed), X, y, cv=LeaveOneOut())
        loocv_acc.append((pred_loocv == y).mean())
        loocv_f1.append(f1_score(y, pred_loocv, average="macro"))
        pred_loto = cross_val_predict(make_rf(seed), X, y, cv=LeaveOneGroupOut(), groups=groups)
        loto_acc.append((pred_loto == y).mean())
        loto_f1.append(f1_score(y, pred_loto, average="macro"))
    print(f"[{name}] {X.shape[1]}f  "
          f"LOOCV {np.mean(loocv_acc) * 100:.1f}±{np.std(loocv_acc) * 100:.1f} F1 {np.mean(loocv_f1):.2f} "
          f"| LOTO {np.mean(loto_acc) * 100:.1f}±{np.std(loto_acc) * 100:.1f} F1 {np.mean(loto_f1):.2f}",
          flush=True)


# Delta|rho| cross-modal coupling features.
X_delta, y_delta, ids_delta, _ = cf.build_feature_matrix(
    [str(_paths.corr_csv(t)) for t in _paths.TRIALS],
    cats=OPERATIONAL_CATS, relabel={"cascade": "pod_crash"})
groups_delta = np.array([trial_of(i) for i in ids_delta])
report("delta", X_delta, y_delta, groups_delta)

# Raw shape-stat features (cached by op_raw_ablation.py).
cached = np.load("/tmp/op_raw_ablation.npz", allow_pickle=True)
X_raw, y_raw, groups_raw = cached["X"], cached["y"], cached["g"]
report("raw", X_raw, y_raw, groups_raw)

# Combined: concatenate raw + delta, aligning the delta rows to the raw row order.
features = pd.concat([pd.read_pickle(f"/tmp/featcache/{t}.pkl") for t in _paths.TRIALS], ignore_index=True)
_, _, ids_raw, _ = ext.build_matrix(features, ext.CATS_OPERATIONAL, ext.RELABEL)
delta_row = {fault_id: i for i, fault_id in enumerate(ids_delta)}
X_combined = np.hstack([X_raw, X_delta[[delta_row[fault_id] for fault_id in ids_raw]]])
report("combined", X_combined, y_raw, groups_raw)
print("[done]", flush=True)
