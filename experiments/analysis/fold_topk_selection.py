"""Fold-wise top-K feature-selection robustness check (leakage-free).

For each leave-one-trial-out fold: rank the raw features by RF importance on the
TRAINING trials only, keep the top K, retrain on those, and predict the held-out
trial. Answers the p>n concern: does the model need all features, or is accuracy
preserved on a small importance-selected subset?
"""

import _paths  # noqa: F401  (anchors sys.path and dataset paths)
from collections import Counter

import numpy as np
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score

cached = np.load("/tmp/op_raw_ablation.npz", allow_pickle=True)
X, y, groups, feature_names = cached["X"], cached["y"], cached["g"], list(cached["feat"])
n_instances, n_features = X.shape
n_trials = len(set(groups))
print(f"n={n_instances} instances, p={n_features} raw features, {n_trials} trials  "
      f"(p>n: {n_features > n_instances})\n")


def feature_modality(col):
    """Modality of a feature column; RTT is grouped with metrics."""
    modality = ext.modality_of(col)
    return "metrics" if modality == "rtt" else modality


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


splitter = LeaveOneGroupOut()
K_VALUES = [10, 20, 30, 50, 100, n_features]
print(f"{'K':>6}{'LOTO acc':>10}{'macroF1':>9}")

selected_at_k20 = Counter()  # how often each feature lands in the top-20, across folds
for K in K_VALUES:
    preds = np.empty(n_instances, dtype=object)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        ranker = make_rf().fit(X[train_idx], y[train_idx])           # importance on TRAIN only
        top_idx = np.argsort(ranker.feature_importances_)[::-1][:K]
        clf = make_rf().fit(X[train_idx][:, top_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx][:, top_idx])
        if K == 20:
            selected_at_k20.update(feature_names[i] for i in top_idx)
    acc = (preds == y).mean()
    f1 = f1_score(y, preds, average="macro")
    print(f"{K:>6}{acc * 100:>9.1f}{f1:>9.2f}{'  (all features)' if K == n_features else ''}")

print(f"\nTop-20 selection stability across the {n_trials} folds:")
always_selected = [feat for feat, count in selected_at_k20.items() if count == n_trials]
print(f"  features selected in ALL {n_trials} folds: {len(always_selected)}")
print(f"  modality mix of those: {dict(Counter(feature_modality(f) for f in always_selected))}")
print("  examples:", ", ".join(always_selected[:8]))
