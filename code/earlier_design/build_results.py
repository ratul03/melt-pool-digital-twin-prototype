"""Numerical diagnostics and exportable research figures; no report generation."""

import os
import json
import numpy as np
import pandas as pd
from common import *

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "verification/matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "svg.fonttype": "none",
    }
)
COLORS = {"ml": "#2563a6", "simple": "#d97732"}
NAMES = {"ml": "ML", "simple": "Training-only simple predictor"}
FIGURES = []


def save(fig, name, source, description):
    fig.savefig(ROOT / f"figures/{name}.png", bbox_inches="tight")
    fig.savefig(ROOT / f"figures/{name}.svg", bbox_inches="tight")
    fig.savefig(ROOT / f"figures/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(
        {
            "id": name,
            "source_tables": source,
            "description": description,
            "png": f"figures/{name}.png",
            "svg": f"figures/{name}.svg",
            "pdf": f"figures/{name}.pdf",
        }
    )


def main():
    m = read_csv(ROOT / "results/conditions.csv")
    a = read_csv(ROOT / "results/per_alloy.csv")
    s = read_csv(ROOT / "results/summary.csv")
    p = read_csv(ROOT / "results/comparison_predictions.csv", dtype={"source": str})
    # Retain zero-survivor cohorts in the audit table rather than silently
    # omitting an alloy whose every test row matches observation-pool inputs.
    groups = ["task", "cohort", "alloy"]
    all_cohorts = p.groupby(groups).row_id.nunique().rename("original_test_rows").reset_index()
    retained_cohorts = read_csv(ROOT / "results/sensitivity_cohorts.csv")[
        groups + ["retained_test_rows"]
    ]
    all_cohorts = all_cohorts.merge(retained_cohorts, on=groups, how="left")
    all_cohorts["retained_test_rows"] = all_cohorts.retained_test_rows.fillna(0).astype(int)
    all_cohorts["included_in_sensitivity_summary"] = all_cohorts.retained_test_rows > 0
    all_cohorts.to_csv(ROOT / "results/sensitivity_cohorts.csv", index=False)
    primary = s[s.cohort == "primary_paired"]
    mp = m[m.cohort == "primary_paired"]
    ap = a[a.cohort == "primary_paired"]
    protocol = json.loads((ROOT / "protocol/supplementary_spec.json").read_text())
    # Empirical variation across dependent selections and model seeds.
    values = ["coverage", "mean_interval_score", "mean_width", "mean_set_size"]
    spread = []
    for keys, g in mp.groupby(["task", "alloy", "budget", "predictor"]):
        for metric in values:
            vals = g[metric].dropna()
            if len(vals):
                spread.append(
                    dict(zip(["task", "alloy", "budget", "predictor"], keys))
                    | dict(
                        metric=metric,
                        min=vals.min(),
                        p10=vals.quantile(0.1),
                        median=vals.median(),
                        p90=vals.quantile(0.9),
                        max=vals.max(),
                        n_conditions=len(vals),
                        interpretation="descriptive dependent seed/selection spread",
                    )
                )
    pd.DataFrame(spread).to_csv(ROOT / "results/condition_spread.csv", index=False)
    # Score change decomposition, including failed alloys.
    changes = []
    for keys, g in a.groupby(["task", "cohort", "alloy", "predictor"]):
        baseline = g[g.budget == 0].iloc[0]
        for _, row in g[g.budget > 0].iterrows():
            item = dict(zip(["task", "cohort", "alloy", "predictor"], keys)) | {
                "budget": int(row.budget)
            }
            for col in [
                "coverage",
                "mean_width",
                "mean_interval_score",
                "lower_miss_penalty",
                "upper_miss_penalty",
                "mean_set_size",
            ]:
                item[col + "_change"] = row[col] - baseline[col]
            changes.append(item)
    pd.DataFrame(changes).to_csv(ROOT / "results/update_changes.csv", index=False)
    # Explicitly hypothetical precision thresholds, with coverage rather than width alone.
    rows = []
    for (task, budget, predictor), g in mp[mp.task != "classification"].groupby(
        ["task", "budget", "predictor"]
    ):
        for tau in protocol["hypothetical_usefulness"]["half_width_tolerances_um"]:
            narrow = (g.mean_width / 2 <= tau).astype(float)
            # Each alloy has the same five seeds x twenty selections.
            fraction = narrow.mean()
            rows.append(
                dict(
                    task=task,
                    budget=budget,
                    predictor=predictor,
                    hypothetical_half_width_um=tau,
                    narrow_state_fraction=fraction,
                    overall_coverage=g.coverage.mean(),
                    covered_and_narrow_fraction=(g.coverage * narrow).mean(),
                    coverage_in_narrow_states=(
                        (g.coverage * narrow).mean() / fraction if fraction else np.nan
                    ),
                    n_alloys=g.alloy.nunique(),
                    n_dependent_states=len(g),
                )
            )
    usefulness = pd.DataFrame(rows)
    usefulness.to_csv(ROOT / "results/hypothetical_precision.csv", index=False)
    # Unique residuals; repeated selections must not multiply the apparent sample size.
    residuals = (
        p[(p.task != "classification") & (p.predictor == "ml") & (p.budget == 0)]
        .drop_duplicates(["key", "row_id"])
        .copy()
    )
    residuals["signed_residual_um"] = residuals.actual.astype(float) - residuals.predicted.astype(
        float
    )
    residuals[
        ["task", "cohort", "alloy", "model_seed", "row_id", "source", "signed_residual_um"]
    ].to_csv(ROOT / "results/signed_residuals.csv", index=False)
    # Data/split counts and source/alloy confounding.
    spec = json.loads((W12 / "protocol/frozen.json").read_text())
    parts = json.loads((W12 / "protocol/splits.json").read_text())
    future = json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
    splits = []
    for t, cfg in spec["tasks"].items():
        d = read_csv(W12 / f"data/{t}.csv", dtype={"source": str}).set_index("row_id")
        for part in [
            x
            for x in parts
            if x["task"] == t and x["seed"] == 0 and x["protocol"] == "leave_one_alloy_out"
        ]:
            fs = [
                f
                for f in future
                if f["task"] == t and f["alloy"] == part["alloy"] and f["repeat"] == 0
            ]
            splits.append(
                dict(
                    task=t,
                    alloy=part["alloy"],
                    usable_task_rows=len(d),
                    seed=0,
                    n_training=len(part["train"]),
                    n_source_calibration=len(part["calibration"]),
                    n_whole_alloy_test=len(part["test"]),
                    n_fixed_final_test=len(fs[0]["test_row_ids"]) if fs else 0,
                    n_target_pool=len(fs[0]["pool_row_ids"]) if fs else 0,
                    paired_eligible=bool(fs and fs[0]["primary_budget_comparison"]),
                    target_source_studies=d.loc[part["test"], "source"].nunique(),
                    features_json=json.dumps(cfg["features"]),
                )
            )
    pd.DataFrame(splits).to_csv(ROOT / "results/data_and_partition_counts.csv", index=False)
    for filename in ["alloy_source_overlap.csv", "exclusions.csv", "feasibility.csv"]:
        (ROOT / "results" / filename).write_bytes((W12 / "data" / filename).read_bytes())

    # Point baseline transfer: original common >=20-record alloy cohort.
    counts = read_csv(W12 / "data/alloy_counts.csv")
    eligible = set(zip(counts.loc[counts.n >= 20, "task"], counts.loc[counts.n >= 20, "alloy"]))
    transfer = []
    for i, part in enumerate(parts):
        key = f"{part['task']}_{part['protocol']}_{i:03d}_s{part['seed']}"
        g = read_csv(W12 / f"results/runs/{key}_predictions.csv")
        for alloy, z in g.groupby("alloy"):
            if (part["task"], alloy) not in eligible:
                continue
            for predictor, column in [("ml", "predicted"), ("simple", "dummy")]:
                row = dict(
                    task=part["task"],
                    protocol=part["protocol"],
                    model_seed=part["seed"],
                    alloy=alloy,
                    predictor=predictor,
                    n_test=len(z),
                )
                if part["task"] == "classification":
                    row["accuracy"] = (z.actual == z[column]).mean()
                else:
                    row["MAE"] = abs(z.actual - z[column]).mean()
                transfer.append(row)
    tr = pd.DataFrame(transfer)
    tr.to_csv(ROOT / "results/transfer_per_alloy_seed.csv", index=False)
    tr_seed = (
        tr.groupby(["task", "protocol", "model_seed", "predictor"])[["MAE", "accuracy"]]
        .mean()
        .reset_index()
    )
    tr_summary = (
        tr_seed.groupby(["task", "protocol", "predictor"])[["MAE", "accuracy"]].mean().reset_index()
    )
    tr_summary = tr_summary.merge(
        tr.groupby(["task", "protocol", "predictor"])
        .alloy.nunique()
        .rename("n_alloys")
        .reset_index(),
        on=["task", "protocol", "predictor"],
    )
    tr_summary.to_csv(ROOT / "results/transfer_summary.csv", index=False)

    # F01: principal paired comparison.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ["mean_interval_score", "coverage", "mean_width"],
        ["Interval score (µm; lower is better)", "Empirical coverage", "Mean interval width (µm)"],
    ):
        for predictor in ["ml", "simple"]:
            z = primary[(primary.task == "depth") & (primary.predictor == predictor)].sort_values(
                "budget"
            )
            ax.plot(z.budget, z[metric], "o-", color=COLORS[predictor], label=NAMES[predictor])
        ax.set(xlabel="Target measurements", ylabel=label, xticks=[0, 9, 20])
        ax.grid(alpha=0.2)
        if metric == "coverage":
            ax.axhline(0.9, color="#555", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1.03)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Depth: paired ML and simple-predictor comparison • 7 alloys • equal alloy weights"
    )
    save(
        fig,
        "01_depth_comparison",
        ["summary.csv"],
        "Same partitions, calibration budgets and fixed test records.",
    )

    # F02: retain all alloys and signs.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained", sharey=True)
    paired = read_csv(ROOT / "results/paired_differences.csv")
    alloys = sorted(ap[ap.task == "depth"].alloy.unique())
    for ax, budget in zip(axes, [0, 9, 20]):
        z = (
            paired[
                (paired.task == "depth")
                & (paired.cohort == "primary_paired")
                & (paired.budget == budget)
            ]
            .set_index("alloy")
            .loc[alloys]
        )
        diff = z.mean_interval_score_ml_minus_simple
        ax.barh(alloys, diff, color=[COLORS["ml"] if x < 0 else COLORS["simple"] for x in diff])
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set(title=f"{budget} target measurements", xlabel="ML − simple interval score (µm)")
    fig.suptitle("Depth by alloy • negative differences favour ML")
    save(
        fig,
        "02_depth_paired_alloys",
        ["paired_differences.csv"],
        "Per-alloy mean differences; no pseudo-replicate significance tests.",
    )

    # F03: explain why score changes despite larger width.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    z = primary[(primary.task == "depth") & (primary.predictor == "ml")].sort_values("budget")
    bottom = np.zeros(len(z))
    for col, color, label in [
        ("mean_width", "#2f6fa3", "Width"),
        ("lower_miss_penalty", "#d6a045", "Below-range penalty"),
        ("upper_miss_penalty", "#b74d4d", "Above-range penalty"),
    ]:
        axes[0].bar(z.budget.astype(str), z[col], bottom=bottom, color=color, label=label)
        bottom += z[col].to_numpy()
    axes[0].set(xlabel="Target measurements", ylabel="Mean interval score components (µm)")
    axes[0].legend(fontsize=8)
    ch = pd.DataFrame(changes)
    z = ch[
        (ch.task == "depth")
        & (ch.cohort == "primary_paired")
        & (ch.predictor == "ml")
        & (ch.budget == 20)
    ].sort_values("alloy")
    axes[1].barh(
        z.alloy,
        z.mean_interval_score_change,
        color=["#2f6fa3" if x < 0 else "#b74d4d" for x in z.mean_interval_score_change],
    )
    axes[1].axvline(0, color="#333", linewidth=0.8)
    axes[1].set(xlabel="20 measurements − source interval score (µm)")
    fig.suptitle("ML depth: width and missed-observation penalties")
    save(
        fig,
        "03_interval_score_decomposition",
        ["summary.csv", "update_changes.csv"],
        "Score = width + lower miss penalty + upper miss penalty.",
    )

    # F04: all outputs; separate scales.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for i, task in enumerate(["depth", "width", "length"]):
        for predictor in ["ml", "simple"]:
            z = primary[(primary.task == task) & (primary.predictor == predictor)].sort_values(
                "budget"
            )
            axes[0, i].plot(
                z.budget, z.coverage, "o-", color=COLORS[predictor], label=NAMES[predictor]
            )
            axes[1, i].plot(z.budget, z.mean_interval_score, "o-", color=COLORS[predictor])
        axes[0, i].axhline(0.9, color="#555", ls="--", lw=1)
        n = int(z.n_alloys.iloc[0])
        axes[0, i].set(
            title=f"{task.title()} • {n} alloys",
            ylabel="Coverage",
            ylim=(0, 1.03),
            xticks=[0, 9, 20],
        )
        axes[1, i].set(
            ylabel="Interval score (µm)", xlabel="Target measurements", xticks=[0, 9, 20]
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Geometry calibration comparisons • paired cohorts")
    save(
        fig,
        "04_geometry_comparison",
        ["summary.csv"],
        "Supporting width and length cohorts remain separate.",
    )

    # F05: classification without claiming coverage/size alone is a proper score.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ["coverage", "mean_set_size", "full_set_fraction"],
        ["Coverage", "Average categories in set", "Fraction containing all 4 categories"],
    ):
        for predictor in ["ml", "simple"]:
            z = primary[
                (primary.task == "classification") & (primary.predictor == predictor)
            ].sort_values("budget")
            ax.plot(z.budget, z[metric], "o-", color=COLORS[predictor], label=NAMES[predictor])
        ax.set(xlabel="Target measurements", ylabel=label, xticks=[0, 9, 20])
    axes[0].axhline(0.9, color="#555", ls="--", lw=1)
    axes[0].set_ylim(0, 1.03)
    axes[1].set_ylim(0, 4.1)
    axes[2].set_ylim(0, 1.03)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Defect classification • 3 paired alloys • simple predictor uses training class frequencies"
    )
    save(
        fig,
        "05_classification_sets",
        ["summary.csv"],
        "Show both empirical coverage and set informativeness.",
    )

    # F06: no confidence intervals inferred from dependent repetitions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", sharey=True)
    for ax, budget in zip(axes, [9, 20]):
        z = mp[(mp.task == "depth") & (mp.predictor == "ml") & (mp.budget == budget)]
        ax.boxplot(
            [z[z.alloy == alloy].mean_interval_score for alloy in alloys],
            vert=False,
            tick_labels=alloys,
            showfliers=True,
            flierprops={"markersize": 2},
        )
        ax.set(xlabel="Interval score (µm)", title=f"{budget} measurements")
    fig.suptitle(
        "Depth: variation over 5 seeds × 20 selections per alloy\nDescriptive dependent repetitions; boxes are not confidence intervals"
    )
    save(
        fig,
        "06_depth_selection_variation",
        ["conditions.csv"],
        "Distribution of condition means, not independent experiments.",
    )

    # F07: signed residuals from unique fixed test records, seed zero.
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    z = residuals[
        (residuals.task == "depth")
        & (residuals.cohort == "primary_paired")
        & (residuals.model_seed == 0)
    ]
    ax.boxplot(
        [z[z.alloy == alloy].signed_residual_um for alloy in alloys],
        vert=False,
        tick_labels=alloys,
        showfliers=True,
    )
    ax.axvline(0, color="#777", ls="--")
    ax.set(
        xlabel="Measured depth − predicted depth (µm)",
        title="Depth residuals • one saved model seed • 20 unique test records per alloy",
    )
    save(
        fig,
        "07_depth_signed_residuals",
        ["signed_residuals.csv"],
        "Seed 0 illustration; all five seeds retained in the data.",
    )

    # F08: overlap sensitivity.
    sens = read_csv(ROOT / "results/sensitivity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, predictor in zip(axes, ["ml", "simple"]):
        for frame, label, style in [
            (primary, "Frozen primary test cohort", "-"),
            (sens[sens.cohort == "primary_paired"], "Matching-input test records removed", "--"),
        ]:
            z = frame[(frame.task == "depth") & (frame.predictor == predictor)].sort_values(
                "budget"
            )
            ax.plot(z.budget, z.mean_interval_score, "o" + style, label=label)
        ax.set(
            title=NAMES[predictor],
            xlabel="Target measurements",
            ylabel="Interval score (µm)",
            xticks=[0, 9, 20],
        )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Depth sensitivity • unchanged models and calibration selections • equal alloy weights"
    )
    save(
        fig,
        "08_overlap_sensitivity",
        ["summary.csv", "sensitivity_summary.csv", "sensitivity_cohorts.csv"],
        "Reduced test cohort is a post-audit sensitivity analysis.",
    )

    # F09: hypothetical usefulness grid, never a manufactured acceptance criterion.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for predictor in ["ml", "simple"]:
        for budget, style in [(0, "--"), (20, "-")]:
            z = usefulness[
                (usefulness.task == "depth")
                & (usefulness.predictor == predictor)
                & (usefulness.budget == budget)
            ]
            label = f"{NAMES[predictor]}, budget {budget}"
            axes[0].plot(
                z.hypothetical_half_width_um,
                z.narrow_state_fraction,
                style,
                color=COLORS[predictor],
                label=label,
            )
            axes[1].plot(
                z.hypothetical_half_width_um,
                z.covered_and_narrow_fraction,
                style,
                color=COLORS[predictor],
            )
    for ax in axes:
        ax.set(xlabel="Hypothetical acceptable half-width (µm)", ylim=(0, 1.03))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction of calibration states within half-width")
    axes[1].set_ylabel("Covered-and-narrow fraction")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Depth precision scenarios • illustrative thresholds, not validated manufacturing tolerances"
    )
    save(
        fig,
        "09_hypothetical_precision",
        ["hypothetical_precision.csv"],
        "State-level widths; no input-specific confidence ranking.",
    )

    # F10: original transfer point comparison on its own cohort.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, task in zip(axes, ["depth", "width", "length"]):
        x = np.arange(2)
        for predictor, offset in [("ml", -0.18), ("simple", 0.18)]:
            z = (
                tr_summary[(tr_summary.task == task) & (tr_summary.predictor == predictor)]
                .set_index("protocol")
                .loc[["random", "leave_one_alloy_out"]]
            )
            ax.bar(x + offset, z.MAE, width=0.35, label=NAMES[predictor], color=COLORS[predictor])
        ax.set(
            title=task.title(),
            xticks=x,
            xticklabels=["Random split", "Excluded alloy"],
            ylabel="Equal-alloy MAE (µm)",
        )
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, tr_summary.loc[tr_summary.task == "depth", "MAE"].max() * 1.35)
    fig.suptitle("Original point-prediction transfer study • common alloys with ≥20 records")
    save(
        fig,
        "10_point_prediction_transfer",
        ["transfer_summary.csv"],
        "Different cohort from the fixed-test calibration study.",
    )
    write_json(ROOT / "figures/index.json", FIGURES)
    print(f"Saved {len(FIGURES)} figures, each as PNG, SVG and PDF.", flush=True)


if __name__ == "__main__":
    main()
