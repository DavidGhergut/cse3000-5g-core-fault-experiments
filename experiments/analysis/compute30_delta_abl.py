"""Delta|rho| modality ablation (operational fig:ablation + security tab:security).

Builds the full Delta|rho| feature matrix once, then restricts it to each
modality-pair channel in memory and reports multi-seed LOOCV (and, for the
operational set, leave-one-trial-out) accuracy.
"""

import re

import _paths  # noqa: F401  (anchors sys.path and dataset paths)
import numpy as np
import classify_faults_rf as cf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

SEEDS = 30


def make_rf(seed):
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def modality_of_signal(signal):
    signal = signal.lower()
    if "log" in signal:
        return "logs"
    if "trace" in signal:
        return "traces"
    return "metrics"


def modality_pair(feature_label):
    """Map a 'sigA → sigB' feature label to its sorted (modality, modality) pair."""
    left, right = feature_label.split("→")
    return tuple(sorted([modality_of_signal(left), modality_of_signal(right)]))


# Each channel = (display name, the set of modality pairs it keeps).
CHANNELS = [
    ("metrics-metrics", {("metrics", "metrics")}),
    ("logs-logs", {("logs", "logs")}),
    ("traces-traces", {("traces", "traces")}),
    ("metrics-logs", {("logs", "metrics")}),
    ("metrics-traces", {("metrics", "traces")}),
    ("logs-traces", {("logs", "traces")}),
    ("ALL cross", {("logs", "metrics"), ("metrics", "traces"), ("logs", "traces")}),
]


def trial_of(fault_id):
    return re.search(r"\[([^\]]+)\]", fault_id).group(1)


def run(paths, cats, relabel, with_loto, title):
    X, y, ids, feature_labels = cf.build_feature_matrix(paths, cats, relabel, cf.MODAL_SETS["all"])
    feature_pairs = [modality_pair(label) for label in feature_labels]
    group_labels = np.array([trial_of(i) for i in ids]) if with_loto else None
    print(f"=== {title} ({X.shape[0]} inst, {X.shape[1]} feat) ===", flush=True)
    for name, allowed_pairs in CHANNELS:
        col_idx = [i for i, pair in enumerate(feature_pairs) if pair in allowed_pairs]
        if not col_idx:
            continue
        loocv, loto = [], []
        for seed in range(SEEDS):
            loocv.append((cross_val_predict(make_rf(seed), X[:, col_idx], y, cv=LeaveOneOut()) == y).mean())
            if with_loto:
                loto.append((cross_val_predict(make_rf(seed), X[:, col_idx], y,
                                                cv=LeaveOneGroupOut(), groups=group_labels) == y).mean())
        line = f"   {name:16}{len(col_idx):4d}f  LOOCV {np.mean(loocv) * 100:5.1f}"
        if with_loto:
            line += f"  LOTO {np.mean(loto) * 100:5.1f}"
        print(line, flush=True)


run([str(_paths.corr_csv(t)) for t in _paths.TRIALS],
    cf.MODE_CONFIGS["operational"]["cats"], cf.MODE_CONFIGS["operational"]["relabel"],
    with_loto=True, title="OPERATIONAL")
run([str(_paths.CORRELATIONS / "security" / "correlations.csv")],
    cf.MODE_CONFIGS["security"]["cats"], cf.MODE_CONFIGS["security"]["relabel"],
    with_loto=False, title="SECURITY")
print("[done]", flush=True)
