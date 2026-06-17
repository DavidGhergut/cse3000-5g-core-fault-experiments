"""Learning curve: does adding more TRIALS (deployments) raise LOTO test accuracy, or plateau?
For each held-out test trial, train on every subset of size k of the remaining trials (exhaustive),
test on the held-out trial. Plot mean test accuracy vs k = number of training deployments.
Two models: raw shape-stat features (headline) and Delta|rho| coupling features.
"""
import sys, os, re, itertools, collections
sys.path.insert(0, "experiments/david")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
import classify_extended_features as ext
import classify_faults_rf as rf
from sklearn.ensemble import RandomForestClassifier

TRIALS = ["boyan","boyan-2","trial4","trial5","trial6","trial7","trial8"]
def trial_of(s): m=re.search(r"\[([^\]]+)\]", s); return m.group(1) if m else s
def mkrf(): return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                          class_weight="balanced", random_state=42, n_jobs=-1)

# ---- RAW features (cache; load per-trial with CORRECT trial tags) ----
CACHE="/tmp/lc_raw.npz"
if os.path.exists(CACHE):
    d=np.load(CACHE,allow_pickle=True); Xr,yr,gr=d["X"],d["y"],d["g"]; print("raw cache loaded",flush=True)
else:
    frames=[]
    for t in TRIALS:
        root=None
        for c in [f"data/5GCore/final_dataset/{t}/C-fault-detection", f"data/5GCore/final_dataset/{t}"]:
            if os.path.isdir(c): root=Path(c); break
        print(f"  loading {t}",flush=True)
        df=ext.load_dataset([root], ext.OPERATIONAL_CATEGORIES, ["mean","std","maxabs","slope"])
        df["dataset"]=t                      # override buggy parent-name tag
        frames.append(df)
    df=pd.concat(frames,ignore_index=True)
    Xr,yr,idr,feat=ext.build_matrix(df, ext.CATS_OPERATIONAL, ext.RELABEL)
    Xr=np.asarray(Xr); yr=np.asarray(yr); gr=np.array([trial_of(s) for s in idr])
    np.savez(CACHE,X=Xr,y=yr,g=gr)
print("RAW trials:",pd.Series(gr).value_counts().to_dict(),flush=True)

# ---- Delta|rho| features (fast, clean trial labels from correlations.csv) ----
paths=[f"data/5GCore/correlations_o5g/{t}/correlations.csv" for t in TRIALS if os.path.exists(f"data/5GCore/correlations_o5g/{t}/correlations.csv")]
cfg=rf.MODE_CONFIGS["operational"]
out=rf.build_feature_matrix(paths, cfg["cats"], cfg["relabel"], rf.MODAL_SETS["cross"])
Xd,yd,idd=np.asarray(out[0]),np.asarray(out[1]),list(out[2])
gd=np.array([trial_of(s) for s in idd])
print("Δ|ρ| trials:",pd.Series(gd).value_counts().to_dict(),flush=True)

def learning_curve(X,y,g):
    trials=sorted(set(g)); curve=collections.defaultdict(list)
    for test in trials:
        te=g==test; others=[tr for tr in trials if tr!=test]
        for k in range(1,len(others)+1):
            for combo in itertools.combinations(others,k):
                trm=np.isin(g,combo)
                clf=mkrf().fit(X[trm],y[trm])
                curve[k].append((clf.predict(X[te])==y[te]).mean())
    ks=sorted(curve); return ks,[np.mean(curve[k])*100 for k in ks],[np.std(curve[k])*100 for k in ks]

print("computing raw learning curve...",flush=True);  kr,mr,sr=learning_curve(Xr,yr,gr)
print("computing Δ|ρ| learning curve...",flush=True);  kd,md,sd=learning_curve(Xd,yd,gd)

print("\n# train trials | RAW acc | Δ|ρ| acc")
for i,k in enumerate(kr):
    print(f"  {k}  |  {mr[i]:.1f}±{sr[i]:.1f}  |  {md[i]:.1f}±{sd[i]:.1f}")

fig,ax=plt.subplots(figsize=(7,4.8))
ax.errorbar(kr,mr,yerr=sr,marker="o",capsize=3,color="#2c6fbb",label="Raw shape-stat features")
ax.errorbar(kd,md,yerr=sd,marker="s",capsize=3,color="#c0392b",label="Δ|ρ| coupling features")
ax.set_xlabel("Number of training deployments (trials)",fontsize=10)
ax.set_ylabel("Held-out-trial test accuracy (%)",fontsize=10)
ax.set_title("Test accuracy vs training deployments: raw still rising, Δ|ρ| plateauing",fontsize=10.5,fontweight="bold")
ax.axhline(32,ls=":",color="grey",lw=1); ax.text(1,33,"majority-class baseline (32%)",fontsize=7,color="grey")
ax.set_xticks(kr); ax.grid(alpha=0.25); ax.legend(fontsize=9,loc="center right")
plt.tight_layout()
out="/Users/david/Desktop/Research paper_david/figures/fig_learning_curve.png"
fig.savefig(out,dpi=150,bbox_inches="tight"); print("\nwrote",out)
print("[done]")
