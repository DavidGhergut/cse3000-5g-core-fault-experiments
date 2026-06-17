import re, types, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score
src=open("experiments/david/cross_correlation.py").read()
TRIALS={"boyan":"data/5GCore/final_dataset/boyan","boyan-2":"data/5GCore/final_dataset/boyan-2",
        "trial4":"data/5GCore/final_dataset/trial4","trial5":"data/5GCore/final_dataset/trial5",
        "trial6":"data/5GCore/final_dataset/trial6/C-fault-detection","trial7":"data/5GCore/final_dataset/trial7/C-fault-detection",
        "trial8":"data/5GCore/final_dataset/trial8/C-fault-detection"}
CROSS=["metrics↔logs","metrics↔traces","traces↔logs"]
def feats_and_acc(df):
    df=df[df.modality_pair.isin(CROSS)].copy(); df["r"]=pd.to_numeric(df.spearman_r,errors="coerce")
    df["cat"]=df.fault_category.replace({"cascade":"pod_crash"})
    piv=df.pivot_table(index=["fault_name","trial","cat","sig_a","sig_b","modality_pair"],columns="window",values="r").reset_index()
    piv=piv.dropna(subset=["pre","during"]); piv["d"]=piv.during.abs()-piv.pre.abs()
    # heatmap metrics-logs per category
    hm={c:piv[(piv.cat==c)&(piv.modality_pair=="metrics↔logs")]["d"].mean() for c in ["infrastructure","network","pfcp","pod_crash"]}
    # classifier matrix
    inst=piv.pivot_table(index=["fault_name","trial","cat"],columns=["sig_a","sig_b"],values="d",fill_value=0.0)
    y=inst.index.get_level_values("cat").values; X=inst.values
    keep=(X!=0).mean(0)>=0.20; X=X[:,keep]
    accs=[accuracy_score(y,cross_val_predict(RandomForestClassifier(n_estimators=200,max_features="sqrt",class_weight="balanced",random_state=s),X,y,cv=LeaveOneOut(),n_jobs=-1)) for s in range(5)]
    return hm, X.shape, np.mean(accs)
print(f"{'bin':>5}{'bins':>6}{'feats':>8}{'LOOCV acc':>11}   heatmap metrics-logs Δ|ρ| (infra/net/pfcp/pod)",flush=True)
for bs in [5,10,15,20,30]:
    nb=300//bs
    s2=re.sub(r"BIN_SEC\s*=\s*10",f"BIN_SEC   = {bs}",src); s2=re.sub(r"N_BINS\s*=\s*30",f"N_BINS    = {nb}",s2)
    mod=types.ModuleType("cc"); mod.__dict__["__name__"]="cc"
    exec(compile(s2,"cc","exec"),mod.__dict__)
    rows=[]
    for tr,dp in TRIALS.items():
        for fd in sorted(Path(dp).iterdir()):
            if fd.is_dir() and fd.name in mod.FAULT_CATEGORIES:
                for r in mod.analyze_fault(fd):
                    r["fault_name"]=fd.name; r["fault_category"]=mod.FAULT_CATEGORIES[fd.name]; r["trial"]=tr; rows.append(r)
    df=pd.DataFrame(rows)
    hm,shape,acc=feats_and_acc(df)
    hmstr="  ".join(f"{hm[c]:+.3f}" for c in ["infrastructure","network","pfcp","pod_crash"])
    print(f"{bs:>4}s{nb:>6}{shape[1]:>8}{acc*100:>10.1f}%     {hmstr}",flush=True)
print("DONE")
