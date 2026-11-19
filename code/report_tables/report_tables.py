"""Report-ready tables and Figure 3.1, built only from frozen result files (2 Nov 2026).
Every value is copied or averaged from a named source file; nothing is re-fitted here."""

import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

W = Path(sys.argv[1])
OUT = Path(sys.argv[2])
REL = W / "Thesis_C_Results_2026-10-20"
W12 = REL / "Thesis_C_Weeks_1_2"
FIN = REL / "Thesis_C_Final_Results"
C22 = W / "Thesis results C 22-10-2026 corrected 2026-11-02"
TBA = W / "tba"
CLS = W / "cls/out"
os.environ.setdefault("MPLCONFIGDIR", str(W / "mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ch3 = OUT / "02_chapter3_benchmark"
ch6 = OUT / "03_chapter6_results"
for p in [ch3 / "tables", ch3 / "figures", ch6 / "tables"]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Table 3.1 models (frozen spec + reference constants)
spec = json.loads((W12 / "protocol/frozen.json").read_text())
const = json.loads((W12 / "inputs/code/reference_constants.json").read_text())
t31 = []
for task, s in spec["tasks"].items():
    setting = {
        "depth": "hidden layers 64-32-64, ReLU, alpha 0.065533, target scaled",
        "width": f"{const['GB_N_ESTIMATORS']['width']} estimators",
        "length": f"{const['GB_N_ESTIMATORS']['length']} estimators",
        "classification": "160 trees",
    }[task]
    t31.append(
        dict(
            output=task,
            model=s["model"],
            usable_rows=s["n"],
            setting=setting,
            features="; ".join(s["features"]),
        )
    )
pd.DataFrame(t31).to_csv(ch3 / "tables/Table_3_1_models.csv", index=False)

# ---------- Table 3.2 reproduction (published values: Akbari et al. 2022 Table 5, from reference_constants PAPER_REFS)
h = pd.read_csv(TBA / "results/historical/summary.csv")
ar = json.loads((TBA / "results/authors_reference/summary.json").read_text())


def hist(m, t, c):
    r = h[(h.model == m) & (h.target == t) & (h.config == c)].iloc[0]
    return r


rows = []
for label, m, t, c, pub in [
    ("Depth NN with absorptivity 2 and hatch spacing", "NN", "depth", "lpbf+abs2+hatch", 0.958),
    ("Width GB with elemental features", "GB", "width", "baseline+elemental", 0.9773),
    ("Length GB with hatch spacing", "GB", "length", "baseline+hatch", 0.9962),
]:
    r = hist(m, t, c)
    rows.append(
        dict(
            configuration=label,
            metric="R2",
            published=pub,
            reproduced_mean=r.R2_mean,
            reproduced_sd=r.R2_std,
            records=int(r.n),
        )
    )
rows.append(
    dict(
        configuration="RFC with both absorptivities and hatch spacing",
        metric="accuracy",
        published=0.8841,
        reproduced_mean=ar["accuracy_mean"],
        reproduced_sd=ar["accuracy_rng_std"],
        records=ar["n"],
    )
)
pd.DataFrame(rows).to_csv(ch3 / "tables/Table_3_2_reproduction.csv", index=False)

# ---------- Table 3.3 transfer, equal alloy weight (20 Oct transfer_summary) + equal-alloy macro-F1 (exact refit of frozen predictions)
ts = pd.read_csv(FIN / "results/transfer_summary.csv")
cs = pd.read_csv(CLS / "transfer_summary.csv")
t33 = []
for task, metric in [
    ("depth", "MAE"),
    ("width", "MAE"),
    ("length", "MAE"),
    ("classification", "accuracy"),
]:
    z = ts[ts.task == task].set_index(["protocol", "predictor"])[metric]
    t33.append(
        dict(
            output=task,
            metric=metric + (" (um)" if metric == "MAE" else ""),
            random_model=z[("random", "ml")],
            random_baseline=z[("random", "simple")],
            unseen_model=z[("leave_one_alloy_out", "ml")],
            unseen_baseline=z[("leave_one_alloy_out", "simple")],
            alloys=int(ts[(ts.task == task)].n_alloys.iloc[0]),
        )
    )
z = cs[cs.version == "original"].set_index(["protocol", "predictor"])["macro_f1"]
t33.append(
    dict(
        output="classification",
        metric="macro-F1",
        random_model=z[("random", "ml")],
        random_baseline=z[("random", "simple")],
        unseen_model=z[("leave_one_alloy_out", "ml")],
        unseen_baseline=z[("leave_one_alloy_out", "simple")],
        alloys=4,
    )
)
t33 = pd.DataFrame(t33)
for side in ["random", "unseen"]:
    err = np.where(
        t33.metric.str.startswith("accuracy"), 1 - t33[side + "_model"], t33[side + "_model"]
    )
    berr = np.where(
        t33.metric.str.startswith("accuracy"), 1 - t33[side + "_baseline"], t33[side + "_baseline"]
    )
    t33[side + "_error_ratio"] = np.where(t33.metric == "macro-F1", np.nan, err / berr)
t33.to_csv(ch3 / "tables/Table_3_3_transfer_equal_alloy.csv", index=False)
# pooled (row-weighted) sensitivity values
p = pd.read_csv(W12 / "results/summary.csv")
p[
    [
        "task",
        "protocol",
        "MAE_mean",
        "dummy_MAE_mean",
        "RMSE_mean",
        "dummy_RMSE_mean",
        "R2_mean",
        "accuracy_mean",
        "dummy_accuracy_mean",
        "macro_f1_mean",
        "dummy_macro_f1_mean",
    ]
].to_csv(ch3 / "tables/Table_3_3_sensitivity_row_weighted.csv", index=False)

# ---------- Figure 3.1 dumbbell: error / baseline error
f = t33[t33.metric != "macro-F1"].copy()
names = {
    "depth": "Depth (MAE)",
    "width": "Width (MAE)",
    "length": "Length (MAE)",
    "classification": "Defect class (1 - accuracy)",
}
INK, INK2, GRID, SURF, BLUE, ORANGE = (
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
    "#ffffff",
    "#2a78d6",
    "#eb6834",
)
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "svg.fonttype": "none",
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    }
)
fig, ax = plt.subplots(figsize=(6.3, 2.6), layout="constrained")
ypos = np.arange(len(f))[::-1]
for y, (_, r) in zip(ypos, f.iterrows()):
    ax.plot(
        [r.random_error_ratio, r.unseen_error_ratio],
        [y, y],
        color=GRID,
        lw=3,
        zorder=1,
        solid_capstyle="round",
    )
    ax.scatter(
        r.random_error_ratio,
        y,
        s=64,
        color=BLUE,
        marker="o",
        edgecolor=SURF,
        linewidth=1.5,
        zorder=3,
    )
    ax.scatter(
        r.unseen_error_ratio,
        y,
        s=64,
        color=ORANGE,
        marker="s",
        edgecolor=SURF,
        linewidth=1.5,
        zorder=3,
    )
    ax.text(
        r.unseen_error_ratio + 0.03,
        y,
        f"{r.unseen_error_ratio:.2f}",
        va="center",
        fontsize=8,
        color=INK2,
    )
    ax.text(
        r.random_error_ratio - 0.03,
        y,
        f"{r.random_error_ratio:.2f}",
        va="center",
        ha="right",
        fontsize=8,
        color=INK2,
    )
ax.axvline(1.0, color=INK2, lw=1, ls="--", zorder=0)
ax.text(1.0, len(f) - 0.45, "baseline", ha="center", fontsize=8, color=INK2)
ax.set_yticks(ypos, [names[t] for t in f.output])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.12, 1.42)
ax.set_xlabel("Model error divided by baseline error (below 1 = better than baseline)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.scatter([], [], s=64, color=BLUE, marker="o", label="Random split")
ax.scatter([], [], s=64, color=ORANGE, marker="s", label="Unseen alloy (leave one alloy out)")
fig.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=8)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(ch3 / f"figures/Figure_3_1_error_ratio.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Chapter 6 tables from the frozen 22 Oct study (clipped, primary cohort)
S = pd.read_csv(C22 / "results/summary.csv")
S = S[S.clipped & (S.cohort == "primary")]
A = pd.read_csv(C22 / "results/per_alloy.csv")
A = A[A.clipped & (A.cohort == "primary")]
M = [
    "source_split",
    "mean_target_split",
    "target_split",
    "target_linPV_jk",
    "ml_shift_jk",
    "ml_auto_jk",
]
LAB = {
    "source_split": "Calibration on other alloys (existing practice)",
    "mean_target_split": "Training mean + target calibration",
    "target_split": "ML + target calibration (20 Oct method)",
    "target_linPV_jk": "Target-only fit on power and speed",
    "ml_shift_jk": "ML + shift update",
    "ml_auto_jk": "ML + automatic update (proposed)",
}
cols = [
    "task",
    "budget",
    "method",
    "coverage",
    "mean_width",
    "interval_score",
    "lower_penalty",
    "upper_penalty",
    "MAE",
    "n_alloys",
    "fraction_conditions_coverage_below_0.8",
]
z = S[S.method.isin(M)][cols].copy()
z["method_label"] = z.method.map(LAB)
z.sort_values(["task", "budget", "method"]).to_csv(
    ch6 / "tables/Table_6_all_methods_all_budgets.csv", index=False
)
z[(z.method == "source_split") & (z.budget == 9)].to_csv(
    ch6 / "tables/Table_6_1_H1_source_calibration.csv", index=False
)
z[(z.task == "depth") & z.method.isin(["source_split", "target_split"])].to_csv(
    ch6 / "tables/Table_6_2_H2_depth_target_calibration.csv", index=False
)
z[z.budget.isin([9, 20])].to_csv(ch6 / "tables/Table_6_3_comparison_budgets_9_20.csv", index=False)
pa = A[A.method.isin(M)][
    ["task", "alloy", "budget", "method", "coverage", "mean_width", "interval_score", "MAE"]
]
pa.to_csv(ch6 / "tables/Table_6_per_alloy.csv", index=False)
# alloy win counts of the proposed method against each comparator (descriptive)
wins = []
for (task, b), g in pa.groupby(["task", "budget"]):
    w = g.pivot(index="alloy", columns="method", values="interval_score")
    for c in M[:-1]:
        wins.append(
            dict(
                task=task,
                budget=b,
                comparator=c,
                comparator_label=LAB[c],
                n_alloys=len(w),
                proposed_better=int((w.ml_auto_jk < w[c]).sum()),
                proposed_IS=w.ml_auto_jk.mean(),
                comparator_IS=w[c].mean(),
            )
        )
pd.DataFrame(wins).to_csv(ch6 / "tables/Table_6_alloy_wins_proposed.csv", index=False)
pc = pd.read_csv(C22 / "results/paired_comparisons.csv")
pc.to_csv(ch6 / "tables/Table_6_4_paired_tests.csv", index=False)
pd.read_csv(C22 / "results/frozen_vs_corrected_summary.csv").to_csv(
    ch6 / "tables/Table_6_frozen_vs_corrected_clipping.csv", index=False
)
pd.read_csv(C22 / "results/auto_choices.csv").to_csv(
    ch6 / "tables/Table_6_4_update_choices.csv", index=False
)
# classification sets: 20 Oct (frozen) and corrected check
fs = pd.read_csv(FIN / "results/summary.csv")
fs = fs[(fs.task == "classification") & (fs.cohort == "primary_paired")]
fs[
    [
        "budget",
        "predictor",
        "coverage",
        "mean_set_size",
        "full_set_fraction",
        "accuracy",
        "macro_f1",
        "n_alloys",
    ]
].to_csv(ch6 / "tables/Table_6_3_classification_sets_frozen.csv", index=False)
pd.read_csv(CLS / "class_sets_summary.csv").to_csv(
    ch6 / "tables/Table_6_3_classification_sets_corrected_check.csv", index=False
)
print("tables written")
print(t33.round(4).to_string())
