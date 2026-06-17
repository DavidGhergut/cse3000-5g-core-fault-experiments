"""Raw modality ablation with accuracy AND macro-F1 (10 seeds), aligned features -> tab:raw_ablation."""
import sys
sys.path.insert(0, "experiments/david")
import numpy as np
import classify_extended_features as ext
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import f1_score

d = np.load("/tmp/op_raw_ablation.npz", allow_pickle=True)
X, y, g, feat = d["X"], d["y"], d["g"], list(d["feat"])
def mof(c):
    m = ext.modality_of(c); return "metrics" if m == "rtt" else m
mod = np.array([mof(c) for c in feat])
def rf(s): return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                         class_weight="balanced", random_state=s, n_jobs=-1)
def ms(ci):
    la, lf, ta, tf = [], [], [], []
    for s in range(10):
        p = cross_val_predict(rf(s), X[:, ci], y, cv=LeaveOneOut())
        la.append((p == y).mean()); lf.append(f1_score(y, p, average="macro"))
        q = cross_val_predict(rf(s), X[:, ci], y, cv=LeaveOneGroupOut(), groups=g)
        ta.append((q == y).mean()); tf.append(f1_score(y, q, average="macro"))
    return np.mean(la) * 100, np.mean(lf), np.mean(ta) * 100, np.mean(tf)

print(f"{'subset':18}{'#f':>4}{'LOOCVacc':>9}{'LOOCVf1':>9}{'LOTOacc':>9}{'LOTOf1':>9}", flush=True)
for s in ["metrics", "logs", "traces", "metrics+logs", "metrics+traces", "logs+traces", "metrics+logs+traces"]:
    w = set(s.split("+")); ci = [i for i, m in enumerate(mod) if m in w]
    la, lf, ta, tf = ms(ci)
    print(f"{('ALL' if s.count('+')==2 else s):18}{len(ci):>4}{la:>9.1f}{lf:>9.2f}{ta:>9.1f}{tf:>9.2f}", flush=True)
print("[done]", flush=True)
