"""Equal-alloy summaries, paired alloy-level comparisons and frozen hypothesis tests."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from common import ROOT, write_json

spec = json.loads((ROOT / "protocol/confirmatory_spec.json").read_text())
m = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results/conditions").glob("*.csv"))], ignore_index=True)
VALS = ["coverage", "mean_width", "interval_score", "lower_penalty", "upper_penalty", "MAE", "negative_lower_fraction"]
K = ["task", "cohort", "alloy", "budget", "method", "clipped"]
seed = m.groupby(K + ["seed"])[VALS].mean().reset_index()
alloy = seed.groupby(K)[VALS].mean().reset_index()
summary = alloy.groupby(["task", "cohort", "budget", "method", "clipped"])[VALS].mean().reset_index()
summary["n_alloys"] = alloy.groupby(["task", "cohort", "budget", "method", "clipped"]).size().to_numpy()
low = m.groupby(["task", "cohort", "budget", "method", "clipped"]).coverage.apply(lambda s: float((s < 0.8).mean()))
summary = summary.merge(low.rename("fraction_conditions_coverage_below_0.8").reset_index())
summary["acceptable_coverage"] = summary.coverage >= 0.85
for name, t in [("per_alloy_seed", seed), ("per_alloy", alloy), ("summary", summary)]:
    t.to_csv(ROOT / f"results/{name}.csv", index=False)

# Paired alloy-level comparisons of the primary proposed method (clipped).
A = alloy[alloy.clipped].set_index(["task", "cohort", "budget", "alloy", "method"]).interval_score.unstack("method")
comparators = ["source_split", "target_split", "mean_target_split", "target_mean_jk", "target_linE_jk",
               "target_linPV_jk", "target_auto_jk", "ml_shift_jk", "ml_linear_jk"]
pairs = []
for (task, cohort, budget), g in A.groupby(level=["task", "cohort", "budget"]):
    for c in comparators:
        diff = (g["ml_auto_jk"] - g[c]).dropna()
        n = len(diff)
        p = float(wilcoxon(diff, method="exact").pvalue) if n >= 5 and (diff != 0).any() else np.nan
        cov = summary[(summary.task == task) & (summary.cohort == cohort) & (summary.budget == budget) & summary.clipped]
        cov = cov.set_index("method").coverage
        pairs.append(dict(task=task, cohort=cohort, budget=budget, comparator=c, n_alloys=n,
                          proposed_IS=float(g["ml_auto_jk"].mean()), comparator_IS=float(g[c].mean()),
                          mean_difference=float(diff.mean()), relative_change=float(diff.mean() / g[c].mean()),
                          alloys_proposed_better=int((diff < 0).sum()), wilcoxon_exact_p=p,
                          proposed_coverage=float(cov["ml_auto_jk"]), comparator_coverage=float(cov[c]),
                          superior_by_frozen_rule=bool(diff.mean() < 0 and (diff < 0).sum() > n / 2
                                                       and cov["ml_auto_jk"] >= 0.85 and cov[c] >= 0.85)))
pairs = pd.DataFrame(pairs)
pairs.to_csv(ROOT / "results/paired_comparisons.csv", index=False)

# Which update did the automatic rules choose (full-data choice)?
choice = m[m.method.isin(["ml_auto_jk", "target_auto_jk"]) & m.clipped].groupby(
    ["task", "cohort", "budget", "method"]).choice.value_counts(normalize=True).rename("fraction").reset_index()
choice.to_csv(ROOT / "results/auto_choices.csv", index=False)

h = pairs[(pairs.task == "depth") & (pairs.cohort == "primary") & (pairs.budget == 9) & (pairs.comparator == "target_split")].iloc[0]
write_json(ROOT / "results/primary_hypothesis.json", dict(
    hypothesis=spec["hypotheses"]["H_primary"], proposed_IS=h.proposed_IS, comparator_IS=h.comparator_IS,
    relative_change=h.relative_change, alloys_proposed_better=int(h.alloys_proposed_better), n_alloys=int(h.n_alloys),
    wilcoxon_exact_p=h.wilcoxon_exact_p, proposed_coverage=h.proposed_coverage, comparator_coverage=h.comparator_coverage,
    supported=bool(h.superior_by_frozen_rule and h.wilcoxon_exact_p < 0.05)))
print(summary[summary.clipped & (summary.cohort == "primary")].pivot_table(
    index=["task", "method"], columns="budget", values="interval_score").round(1).to_string())
print(pairs[(pairs.cohort == "primary")][["task", "budget", "comparator", "proposed_IS", "comparator_IS",
      "alloys_proposed_better", "n_alloys", "wilcoxon_exact_p", "superior_by_frozen_rule"]].round(4).to_string(index=False))
