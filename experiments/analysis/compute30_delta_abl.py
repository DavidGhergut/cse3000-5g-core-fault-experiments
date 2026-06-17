"""30-seed Delta|rho| modality ablation (operational fig:ablation + security tab:security),
on Open5GS-integrated correlations. Build modal='all' ONCE, subset channels in-memory."""
import sys, re
sys.path.insert(0, "experiments/david")
import numpy as np
import classify_faults_rf as cf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict

TRIALS = ["boyan", "boyan-2", "trial4", "trial5", "trial6", "trial7", "trial8"]
SEEDS = 30
def rf(s): return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                         class_weight="balanced", random_state=s, n_jobs=-1)
def side(s):
    s = s.lower()
    return "logs" if "log" in s else ("traces" if "trace" in s else "metrics")
def pair(f):
    a, b = f.split("→"); return tuple(sorted([side(a), side(b)]))
CH = [("metrics-metrics", {("metrics", "metrics")}), ("logs-logs", {("logs", "logs")}),
      ("traces-traces", {("traces", "traces")}), ("metrics-logs", {("logs", "metrics")}),
      ("metrics-traces", {("metrics", "traces")}), ("logs-traces", {("logs", "traces")}),
      ("ALL cross", {("logs", "metrics"), ("metrics", "traces"), ("logs", "traces")})]

def run(paths, cats, relabel, groups, title):
    X, y, ids, feat = cf.build_feature_matrix(paths, cats, relabel, cf.MODAL_SETS["all"])
    pr = [pair(f) for f in feat]
    g = np.array([re.search(r"\[([^\]]+)\]", s).group(1) for s in ids]) if groups else None
    print(f"=== {title} ({X.shape[0]} inst, {X.shape[1]} feat) ===", flush=True)
    for name, keep in CH:
        ci = [i for i, p in enumerate(pr) if p in keep]
        if not ci: continue
        la, lt = [], []
        for s in range(SEEDS):
            la.append((cross_val_predict(rf(s), X[:, ci], y, cv=LeaveOneOut()) == y).mean())
            if groups:
                lt.append((cross_val_predict(rf(s), X[:, ci], y, cv=LeaveOneGroupOut(), groups=g) == y).mean())
        out = f"   {name:16}{len(ci):4d}f  LOOCV {np.mean(la)*100:5.1f}"
        if groups: out += f"  LOTO {np.mean(lt)*100:5.1f}"
        print(out, flush=True)

run([f"data/5GCore/correlations_o5g/{t}/correlations.csv" for t in TRIALS],
    cf.MODE_CONFIGS["operational"]["cats"], cf.MODE_CONFIGS["operational"]["relabel"], True, "OPERATIONAL")
run(["data/5GCore/correlations_o5g/security/correlations.csv"],
    cf.MODE_CONFIGS["security"]["cats"], cf.MODE_CONFIGS["security"]["relabel"], False, "SECURITY")
print("[done]", flush=True)
