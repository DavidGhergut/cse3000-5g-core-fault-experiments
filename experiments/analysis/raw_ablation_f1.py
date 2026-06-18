"""Raw modality ablation with accuracy AND macro-F1, averaged over 10 seeds.

Reads the cached raw feature matrix (built by op_raw_ablation.py) and reports
per-modality-subset LOOCV and leave-one-trial-out (LOTO) accuracy and macro-F1.
Produces tab:raw_ablation.
"""

import _paths  # noqa: F401  (anchors sys.path and dataset paths)
import numpy as np
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import f1_score

cached = np.load("/tmp/op_raw_ablation.npz", allow_pickle=True)
X, y, groups, feature_names = cached["X"], cached["y"], cached["g"], list(cached["feat"])


def feature_modality(col):
    """Modality of a feature column; RTT is grouped with metrics."""
    modality = ext.modality_of(col)
    return "metrics" if modality == "rtt" else modality


feature_mods = np.array([feature_modality(col) for col in feature_names])


def make_rf(seed):
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def evaluate(col_idx):
    """Mean accuracy/F1 over 10 seeds, for both LOOCV and LOTO, on the given columns."""
    loocv_acc, loocv_f1, loto_acc, loto_f1 = [], [], [], []
    for seed in range(10):
        pred_loocv = cross_val_predict(make_rf(seed), X[:, col_idx], y, cv=LeaveOneOut())
        loocv_acc.append((pred_loocv == y).mean())
        loocv_f1.append(f1_score(y, pred_loocv, average="macro"))
        pred_loto = cross_val_predict(make_rf(seed), X[:, col_idx], y, cv=LeaveOneGroupOut(), groups=groups)
        loto_acc.append((pred_loto == y).mean())
        loto_f1.append(f1_score(y, pred_loto, average="macro"))
    return np.mean(loocv_acc) * 100, np.mean(loocv_f1), np.mean(loto_acc) * 100, np.mean(loto_f1)


SUBSETS = ["metrics", "logs", "traces", "metrics+logs", "metrics+traces",
           "logs+traces", "metrics+logs+traces"]
print(f"{'subset':18}{'#f':>4}{'LOOCVacc':>9}{'LOOCVf1':>9}{'LOTOacc':>9}{'LOTOf1':>9}", flush=True)
for subset in SUBSETS:
    wanted = set(subset.split("+"))
    col_idx = [i for i, modality in enumerate(feature_mods) if modality in wanted]
    acc_loocv, f1_loocv, acc_loto, f1_loto = evaluate(col_idx)
    label = "ALL" if subset.count("+") == 2 else subset
    print(f"{label:18}{len(col_idx):>4}{acc_loocv:>9.1f}{f1_loocv:>9.2f}{acc_loto:>9.1f}{f1_loto:>9.2f}", flush=True)
print("[done]", flush=True)
