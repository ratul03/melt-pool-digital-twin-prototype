"""Figures and one post-freeze descriptive table. No new tests or method choices.

Colour: validated categorical palette (dataviz reference, light mode), fixed slot per method.
Three slots sit below 3:1 contrast, so every series also has its own marker and a direct label.
"""

import os
import itertools
import numpy as np
import pandas as pd
from common import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "verification/mpl_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [  # fixed slot order; colour follows the method, never its rank
    ("ml_auto_jk", "Proposed: ML + update (auto)", "#2a78d6", "o"),
    ("target_split", "20 Oct method: ML + target calibration", "#eb6834", "s"),
    ("target_linPV_jk", "Target-only fit on P and V", "#1baf7a", "^"),
    ("mean_target_split", "Training mean + target calibration", "#eda100", "D"),
    ("ml_shift_jk", "ML + shift update", "#e87ba4", "v"),
]
SHORT = {
    "ml_auto_jk": "Proposed",
    "target_split": "20 Oct method",
    "target_linPV_jk": "Target-only P+V",
    "mean_target_split": "Training mean",
    "ml_shift_jk": "ML shift",
}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "svg.fonttype": "none",
    }
)
S = pd.read_csv(ROOT / "results/summary.csv")
S = S[S.clipped & (S.cohort == "primary")]
A = pd.read_csv(ROOT / "results/per_alloy.csv")
A = A[A.clipped & (A.cohort == "primary")]
(ROOT / "figures").mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(ROOT / f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def curves(metric, ylabel, name, ref=None):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), layout="constrained")
    for ax, task in zip(axes, ["depth", "width", "length"]):
        z = S[S.task == task]
        for m, label, color, marker in SERIES:
            g = z[z.method == m].sort_values("budget")
            ax.plot(
                g.budget,
                g[metric],
                color=color,
                lw=2,
                marker=marker,
                ms=8,
                mec=SURF,
                mew=1.5,
                label=label,
            )
            if task == "depth" and metric == "interval_score":
                ax.annotate(
                    SHORT[m],
                    (g.budget.iloc[-1], g[metric].iloc[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7.5,
                    color=INK2,
                )
        src = z[z.method == "source_split"][metric]
        n = int(z.n_alloys.max())
        note = (
            f"Source calibration: {src.iloc[0]:.0f} µm (off scale)"
            if metric == "interval_score"
            else f"Source calibration coverage: {src.iloc[0]:.2f} (off scale)"
        )
        ax.set_title(f"{task.title()} · {n} alloys\n{note}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Target-alloy measurements")
        bud = sorted(z.budget.unique())
        cnt = z[z.method == "ml_auto_jk"].set_index("budget").n_alloys
        ax.set_xticks(bud, [f"{b}" if cnt[b] == n else f"{b}\n({cnt[b]} alloys)" for b in bud])
        ax.grid(axis="y", color=GRID, lw=0.8)
        if ref:
            for r, ls in ref:
                ax.axhline(r, color=INK2, ls=ls, lw=1)
    axes[0].set_ylabel(ylabel)
    if metric == "interval_score":
        axes[0].set_xlim(right=axes[0].get_xlim()[1] + 6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    save(fig, name)


curves(
    "interval_score",
    "Interval score, 90% (µm, lower is better)",
    "F1_interval_score_vs_measurements",
)
curves(
    "coverage", "Empirical coverage", "F2_coverage_vs_measurements", ref=[(0.9, "--"), (0.85, ":")]
)

# F3: per-alloy depth, three key methods, log scale (alloys differ by 40x)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
alloys = sorted(A[A.task == "depth"].alloy.unique())
for ax, b in zip(axes, [9, 20]):
    z = A[(A.task == "depth") & (A.budget == b)].set_index(["alloy", "method"]).interval_score
    for i, alloy in enumerate(alloys):
        vals = [z[(alloy, m)] for m, *_ in SERIES[:3]]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, zorder=1)
    for k, (m, label, color, marker) in enumerate(SERIES[:3]):
        ax.scatter(
            [z[(a, m)] for a in alloys],
            np.arange(len(alloys)) + (k - 1) * 0.12,
            color=color,
            marker=marker,
            s=60,
            edgecolor=SURF,
            linewidth=1.5,
            zorder=3,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yticks(range(len(alloys)), alloys)
    ax.set_title(f"Depth · {b} target measurements · each alloy", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Interval score, 90% (µm, log scale)")
    ax.grid(axis="x", color=GRID, lw=0.8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
save(fig, "F3_depth_per_alloy")

# Post-freeze DESCRIPTIVE table: alloy win counts for every pair of methods (no tests).
B = pd.read_csv(ROOT / "results/per_alloy.csv")
B = B[B.clipped & (B.cohort == "primary")]
rows = []
for (task, budget), g in B.groupby(["task", "budget"]):
    w = g.pivot(index="alloy", columns="method", values="interval_score")
    for a, b in itertools.permutations(w.columns, 2):
        rows.append(
            dict(
                task=task,
                budget=budget,
                method=a,
                versus=b,
                alloys_better=int((w[a] < w[b]).sum()),
                n_alloys=len(w),
                mean_IS=w[a].mean(),
                versus_mean_IS=w[b].mean(),
                median_IS=w[a].median(),
                versus_median_IS=w[b].median(),
            )
        )
pd.DataFrame(rows).to_csv(ROOT / "results/descriptive_pairwise_wins.csv", index=False)
print("figures and descriptive table written")
