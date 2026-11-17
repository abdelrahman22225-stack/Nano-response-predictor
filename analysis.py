"""
Nano-response predictor
=======================

Which bacteria will a chitosan Schiff base nanoparticle affect, and at what dose?

Data: Soliman, Bayoumy et al., Scientific Reports 16, 24305 (2026),
Table 1 (inhibition zones) and Table 3 (biofilm inhibition).

This is a small learning project. There are only 9 strains, so every result is
exploratory. The point is to practise an honest workflow: a baseline first,
testing on strains the model has never seen, and saying clearly what the data
cannot tell us.

Run:  python analysis.py      (figures are saved in the figures/ folder)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw figures without a screen
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
zones = pd.read_csv("data/inhibition_zones.csv")
biofilm = pd.read_csv("data/biofilm_inhibition.csv")
strains = pd.read_csv("data/strains.csv")

print("Strains:", len(strains))
print("Inhibition-zone measurements:", len(zones))
print("Biofilm measurements:", len(biofilm))

# ---------------------------------------------------------------------------
# 2. Look at the data first: a heat map of nanoparticle zones
# ---------------------------------------------------------------------------
np_zones = zones[zones.formulation == "nanoparticle"]
table = np_zones.pivot(index="strain", columns="conc_mg_per_ml", values="zone_mm")
table = table.loc[strains.strain]          # keep the paper's strain order

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=30)
ax.set_xticks(range(3), [f"{c} mg/mL" for c in table.columns])
ax.set_yticks(range(len(table)), table.index)
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        v = table.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if v > 18 else "black", fontsize=9)
fig.colorbar(im, label="Inhibition zone (mm)")
ax.set_title("Nanoparticle inhibition zones")
fig.tight_layout()
fig.savefig("figures/1_zone_heatmap.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Dose-response: how fast does each strain respond as the dose goes up?
#    Fit  zone = a + b * log10(concentration)  for every strain.
#    b (the slope) = extra millimetres of zone for every 10-fold more particle.
# ---------------------------------------------------------------------------
rows = []
for s in strains.strain:
    d = np_zones[np_zones.strain == s]
    x = np.log10(d.conc_mg_per_ml.values).reshape(-1, 1)
    y = d.zone_mm.values
    fit = LinearRegression().fit(x, y)
    rows.append({"strain": s,
                 "zone_at_10_mg_per_ml": y[-1],
                 "slope_mm_per_10fold": fit.coef_[0]})
dose = pd.DataFrame(rows).merge(strains, on="strain")
dose = dose.sort_values("zone_at_10_mg_per_ml", ascending=False)
print("\nStrains ranked by nanoparticle zone at 10 mg/mL:")
print(dose[["strain", "zone_at_10_mg_per_ml", "slope_mm_per_10fold"]]
      .round(1).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. The prediction question
#    If we meet a NEW strain, can we predict its nanoparticle zone from cheap
#    information: Gram type, biofilm former or not, the dose, and the zone the
#    parent compound gives at the same dose?
#
#    Honest test = leave-one-strain-out: hide one strain, train on the other 8,
#    predict the hidden one, repeat for all 9 strains.
# ---------------------------------------------------------------------------
comp = zones[zones.formulation == "compound"][["strain", "conc_mg_per_ml", "zone_mm"]]
comp = comp.rename(columns={"zone_mm": "compound_zone_mm"})
data = (np_zones.merge(comp, on=["strain", "conc_mg_per_ml"])
                .merge(strains, on="strain"))
data["log10_conc"] = np.log10(data.conc_mg_per_ml)

FEATURES = ["log10_conc", "gram_positive", "biofilm_former_cra", "compound_zone_mm"]
TARGET = "zone_mm"

predictions = {"baseline (dose only)": [], "linear regression": [], "small decision tree": []}
truth, who = [], []

for hidden in strains.strain:
    train = data[data.strain != hidden]
    test = data[data.strain == hidden]

    # Baseline: average zone of the other strains at the same dose
    mean_by_dose = train.groupby("conc_mg_per_ml")[TARGET].mean()
    predictions["baseline (dose only)"] += list(test.conc_mg_per_ml.map(mean_by_dose))

    lin = LinearRegression().fit(train[FEATURES], train[TARGET])
    predictions["linear regression"] += list(lin.predict(test[FEATURES]))

    tree = DecisionTreeRegressor(max_depth=2, random_state=0)
    tree.fit(train[FEATURES], train[TARGET])
    predictions["small decision tree"] += list(tree.predict(test[FEATURES]))

    truth += list(test[TARGET])
    who += list(test.strain)

print("\nLeave-one-strain-out error (mean absolute error, mm):")
results = {}
for name, pred in predictions.items():
    results[name] = mean_absolute_error(truth, pred)
    print(f"  {name:22s} {results[name]:.2f} mm")

# What did the linear model learn? (fitted on all 9 strains, for reading only)
lin_all = LinearRegression().fit(data[FEATURES], data[TARGET])
print("\nLinear model weights (all strains):")
for f, w in zip(FEATURES, lin_all.coef_):
    print(f"  {f:20s} {w:+.2f}")

# Plot: predicted vs measured for the best model
best = min(results, key=results.get)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(truth, predictions["baseline (dose only)"], label="baseline", alpha=0.6, marker="x")
ax.scatter(truth, predictions[best], label=best, alpha=0.8)
ax.plot([0, 30], [0, 30], "k--", lw=1)
ax.set_xlabel("Measured zone (mm)")
ax.set_ylabel("Predicted zone (mm), strain hidden during training")
ax.set_title("Predicting unseen strains")
ax.legend()
fig.tight_layout()
fig.savefig("figures/2_predicted_vs_measured.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Biofilm: what fraction of the MIC is needed for 50% inhibition?
#    Linear interpolation between the three tested doses (0.25, 0.5, 1 x MIC).
#    Negative values mean MORE biofilm than the untreated control.
# ---------------------------------------------------------------------------
def dose_for(target, x, y):
    """Smallest tested-range dose where inhibition reaches `target` %."""
    for i in range(len(x) - 1):
        if y[i] < target <= y[i + 1]:
            return x[i] + (target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return x[0] if y[0] >= target else np.nan   # nan = not reached

rows = []
for (s, form), d in biofilm.groupby(["strain", "formulation"]):
    d = d.sort_values("fraction_of_mic")
    x, y = d.fraction_of_mic.values, d.inhibition_pct.values
    rows.append({"strain": s, "formulation": form,
                 "dose_for_50pct_xMIC": dose_for(50, x, y),
                 "enhances_biofilm_below_MIC": bool((y < 0).any())})
bio = pd.DataFrame(rows)
print("\nBiofilm: fraction of MIC needed for 50% inhibition")
print(bio.pivot(index="strain", columns="formulation", values="dose_for_50pct_xMIC")
         .round(2).to_string())
print("\nStrains where a sub-MIC dose INCREASED biofilm:")
print(bio[bio.enhances_biofilm_below_MIC][["strain", "formulation"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
for s, d in biofilm[biofilm.formulation == "nanoparticle"].groupby("strain"):
    ax.plot(d.fraction_of_mic, d.inhibition_pct, marker="o", label=s)
ax.axhline(0, color="grey", lw=1)
ax.axhline(50, color="grey", lw=1, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks([0.25, 0.5, 1], ["0.25x", "0.5x", "1x MIC"])
ax.set_ylabel("Biofilm inhibition (%)")
ax.set_title("Nanoparticles: biofilm inhibition rises with dose")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("figures/3_biofilm_dose_response.png", dpi=150)
plt.close(fig)

print("\nDone. Figures saved in figures/.")
