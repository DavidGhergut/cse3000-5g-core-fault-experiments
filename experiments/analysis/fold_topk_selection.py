"""Fold-wise top-K feature-selection robustness check (leakage-free).
For each leave-one-trial-out fold: rank raw features by RF importance on the
TRAINING trials only, keep the top K, retrain on those, predict the held-out
trial. Answers the examiner's p>n question: does the model need all 215
features, or is accuracy preserved on a small importance-selected subset?"""
import sys
sys.path.insert(0, "experiments/david")
import numpy as np
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from collections import Counter

d = np.load("/tmp/op_raw_ablation.npz", allow_pickle=True)
X, y, g, feat = d["X"], d["y"], d["g"], list(d["feat"])
n, p = X.shape
print(f"n={n} instances, p={p} raw features, {len(set(g))} trials  (p>n: {p>n})\n")


def mof(c):
    m = ext.modality_of(c); return "metrics" if m == "rtt" else m


mod = np.array([mof(c) for c in feat])
def mkrf(): return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                          class_weight="balanced", random_state=42, n_jobs=-1)


logo = LeaveOneGroupOut()
Ks = [10, 20, 30, 50, 100, p]
print(f"{'K':>6}{'LOTO acc':>10}{'macroF1':>9}")
core20 = Counter()
for K in Ks:
    preds = np.empty(n, dtype=object)
    for tr, te in logo.split(X, y, groups=g):
        ranker = mkrf().fit(X[tr], y[tr])                       # importance on TRAIN only
        idx = np.argsort(ranker.feature_importances_)[::-1][:K]
        clf = mkrf().fit(X[tr][:, idx], y[tr])
        preds[te] = clf.predict(X[te][:, idx])
        if K == 20:
            core20.update(feat[i] for i in idx)
    acc = (preds == y).mean(); f1 = f1_score(y, preds, average="macro")
    print(f"{K:>6}{acc*100:>9.1f}{f1:>9.2f}{'  (all features)' if K == p else ''}")

print("\nTop-20 selection stability across the 7 folds:")
stable = [f for f, c in core20.items() if c == 7]
print(f"  features selected in ALL 7 folds: {len(stable)}")
print(f"  modality mix of those: {dict(Counter(mof(f) for f in stable))}")
print("  examples:", ", ".join(stable[:8]))
