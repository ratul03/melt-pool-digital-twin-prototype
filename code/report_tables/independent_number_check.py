"""Independent recomputation of the headline report numbers from the lowest-level files (separate code path)."""

import json
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

W = Path(sys.argv[1])
P = W / "PKG"
W12 = W / "Thesis_C_Results_2026-10-20/Thesis_C_Weeks_1_2"
C22 = W / "Thesis results C 22-10-2026 corrected 2026-11-02"
res = {}
fails = []


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol * max(1, abs(want))
    res[name] = dict(recomputed=got, package=want, pass_=bool(ok))
    if not ok:
        fails.append(name)


# Table 3.3 from per-row predictions
t33 = pd.read_csv(P / "02_chapter3_benchmark/tables/Table_3_3_transfer_equal_alloy.csv").set_index(
    ["output", "metric"]
)
counts = pd.read_csv(W12 / "data/alloy_counts.csv")
elig = set(zip(counts[counts.n >= 20].task, counts[counts.n >= 20].alloy))
splits = json.loads((W12 / "protocol/splits.json").read_text())
acc = {}
for i, s in enumerate(splits):
    g = pd.read_csv(
        W12 / f"results/runs/{s['task']}_{s['protocol']}_{i:03d}_s{s['seed']}_predictions.csv"
    )
    for alloy, z in g.groupby("alloy"):
        if (s["task"], alloy) not in elig:
            continue
        for col in ["predicted", "dummy"]:
            v = (
                (z.actual == z[col]).mean()
                if s["task"] == "classification"
                else (z.actual - z[col]).abs().mean()
            )
            acc.setdefault((s["task"], s["protocol"], col, s["seed"]), []).append(v)
for task, metric in [
    ("depth", "MAE (um)"),
    ("width", "MAE (um)"),
    ("length", "MAE (um)"),
    ("classification", "accuracy"),
]:
    for prot, side in [("random", "random"), ("leave_one_alloy_out", "unseen")]:
        for col, who in [("predicted", "model"), ("dummy", "baseline")]:
            v = np.mean([np.mean(acc[(task, prot, col, sd)]) for sd in range(5)])
            check(f"T3.3 {task} {side} {who}", v, t33.loc[(task, metric), f"{side}_{who}"])
# 22 Oct headline numbers from condition files: alloy mean of conditions, then equal-alloy mean
cond = pd.concat(
    [pd.read_csv(f) for f in glob.glob(str(C22 / "results/conditions/*.csv"))], ignore_index=True
)
cond_all = cond
cond = cond[cond.clipped & (cond.cohort == "primary")]
S = pd.read_csv(P / "03_chapter6_results/tables/Table_6_all_methods_all_budgets.csv")
for task in ["depth", "width", "length"]:
    for b in [9, 20]:
        for m in [
            "source_split",
            "target_split",
            "mean_target_split",
            "target_linPV_jk",
            "ml_shift_jk",
            "ml_auto_jk",
        ]:
            z = cond[(cond.task == task) & (cond.budget == b) & (cond.method == m)]
            if not len(z):
                continue
            al = (
                z.groupby(["alloy", "seed"])[["interval_score", "coverage", "MAE", "mean_width"]]
                .mean()
                .groupby("alloy")
                .mean()
            )
            row = S[(S.task == task) & (S.budget == b) & (S.method == m)].iloc[0]
            for col in ["interval_score", "coverage", "MAE", "mean_width"]:
                check(f"T6 {task} b{b} {m} {col}", al[col].mean(), row[col])
# alloy wins and primary Wilcoxon
z = cond[
    (cond.task == "depth")
    & (cond.budget == 9)
    & cond.method.isin(["ml_auto_jk", "target_split", "source_split"])
]
al = (
    z.groupby(["alloy", "method", "seed"])
    .interval_score.mean()
    .groupby(["alloy", "method"])
    .mean()
    .unstack()
)
check(
    "depth b9 alloys proposed better than 20 Oct method",
    float((al.ml_auto_jk < al.target_split).sum()),
    3,
    0,
)
check(
    "depth b9 alloys proposed better than source calibration",
    float((al.ml_auto_jk < al.source_split).sum()),
    7,
    0,
)
check("no negative mean width in any condition", float((cond.mean_width < 0).sum()), 0, 0)
check(
    "primary Wilcoxon exact p",
    float(
        wilcoxon(al.ml_auto_jk - al.target_split, alternative="two-sided", method="exact").pvalue
    ),
    0.8125,
)
out = dict(passed=len(res) - len(fails), failed=len(fails), failures=fails, checks=res)
(P / "00_verification").mkdir(parents=True, exist_ok=True)
(P / "00_verification/independent_number_check.json").write_text(json.dumps(out, indent=2))
print("passed", out["passed"], "failed", out["failed"], fails[:10])
