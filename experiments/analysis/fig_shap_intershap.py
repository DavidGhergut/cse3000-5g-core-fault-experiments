#!/usr/bin/env python3
"""Two attribution lenses on cross-modal importance (values from the existing
fig_shap_vs_intershap analysis): SHAP importance on the Delta|rho| model vs.
SHAP interaction (InterSHAP) on the raw model. They rank the three pairs in
opposite order, illustrating characterization (coupling discriminability) vs.
classification (raw-feature interaction)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

pairs = ["metrics-\nlogs", "metrics-\ntraces", "logs-\ntraces"]
col   = ["#c0392b", "#e8833a", "#3b7cc4"]            # M-L, M-T, L-T

# left: Delta|rho| SHAP importance (mean |SHAP| per feature) -- ALIGNED features, 154 inst
dl   = [0.0026, 0.0017, 0.0040]; dl_e = [0, 0, 0]
# right: raw InterSHAP (mean |SHAP interaction|, x10^-3) -- ALIGNED features, 154 inst
ir   = [0.008, 0.006, 0.007];    ir_e = [0, 0, 0]

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.8))
fig.suptitle("SHAP and InterSHAP emphasize different cross-modal pairs",
             fontsize=11.5, fontweight="bold", y=1.02)

ax[0].bar(pairs, dl, yerr=dl_e, color=col, capsize=3, edgecolor="black", linewidth=0.4)
for i,v in enumerate(dl): ax[0].text(i, v+dl_e[i]+0.00015, f"{v:.4f}", ha="center", fontsize=8)
ax[0].set_title(r"$\Delta|\rho|$ model — SHAP importance"+"\n(which coupling change discriminates faults)", fontsize=9)
ax[0].set_ylabel("mean |SHAP| per feature", fontsize=9); ax[0].set_ylim(0, 0.0050)

ax[1].bar(pairs, ir, yerr=ir_e, color=col, capsize=3, edgecolor="black", linewidth=0.4)
for i,v in enumerate(ir): ax[1].text(i, v+ir_e[i]+0.0006, f"{v:.3f}", ha="center", fontsize=8)
ax[1].set_title("Raw model — InterSHAP interaction\n(which signal pair the classifier exploits jointly)", fontsize=9)
ax[1].set_ylabel(r"mean SHAP interaction ($\times10^{-3}$)", fontsize=9); ax[1].set_ylim(0, 0.011)

for a in ax: a.tick_params(axis="x", labelsize=8.5); a.grid(axis="y", alpha=0.25)
plt.tight_layout()
out = "/Users/david/Desktop/Research paper_david/figures/fig_shap_intershap.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
