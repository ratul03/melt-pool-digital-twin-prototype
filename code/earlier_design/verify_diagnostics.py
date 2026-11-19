"""Checks of derived diagnostic tables and figure inputs."""

import json
import numpy as np
import pandas as pd
from common import ROOT, W12, W34, read_csv, write_json

checks = []


def check(name, value):
    checks.append({"check": name, "passed": bool(value)})


def near(a, b):
    return np.allclose(a, b, rtol=1e-10, atol=1e-8, equal_nan=True)


m = read_csv(ROOT / "results/conditions.csv")
s = read_csv(ROOT / "results/summary.csv")
a = read_csv(ROOT / "results/per_alloy.csv")
geometry = s[s.task != "classification"]
check(
    "interval_score_decomposition",
    near(
        geometry.mean_interval_score,
        geometry.mean_width + geometry.lower_miss_penalty + geometry.upper_miss_penalty,
    ),
)
changes = read_csv(ROOT / "results/update_changes.csv")
for _, row in changes.iterrows():
    subset = a[
        (a.task == row.task)
        & (a.cohort == row.cohort)
        & (a.alloy == row.alloy)
        & (a.predictor == row.predictor)
    ].set_index("budget")
    for col in [c[:-7] for c in changes.columns if c.endswith("_change")]:
        check(
            "paired_update_delta",
            near(row[col + "_change"], subset.loc[row.budget, col] - subset.loc[0, col]),
        )
precision = read_csv(ROOT / "results/hypothetical_precision.csv")
for _, row in precision.iterrows():
    q = m[
        (m.task == row.task)
        & (m.budget == row.budget)
        & (m.predictor == row.predictor)
        & (m.cohort == "primary_paired")
    ]
    narrow = q.mean_width.to_numpy() / 2 <= row.hypothetical_half_width_um
    check("hypothetical_state_fraction", near(narrow.mean(), row.narrow_state_fraction))
    check(
        "hypothetical_joint_coverage",
        near((narrow * q.coverage.to_numpy()).mean(), row.covered_and_narrow_fraction),
    )
    check(
        "hypothetical_conditional_coverage",
        near(
            q.loc[narrow, "coverage"].mean() if narrow.any() else np.nan,
            row.coverage_in_narrow_states,
        ),
    )
for _, g in precision.groupby(["task", "budget", "predictor"]):
    g = g.sort_values("hypothetical_half_width_um")
    check(
        "precision_grid_monotone",
        (np.diff(g.narrow_state_fraction) >= 0).all()
        and (np.diff(g.covered_and_narrow_fraction) >= -1e-12).all(),
    )
new_transfer = read_csv(ROOT / "results/transfer_per_alloy_seed.csv")
old_transfer = read_csv(W12 / "results/primary_per_alloy_per_seed.csv")
match = new_transfer[new_transfer.predictor == "ml"].merge(
    old_transfer,
    left_on=["task", "protocol", "alloy", "model_seed"],
    right_on=["task", "protocol", "alloy", "seed"],
    suffixes=("_new", "_old"),
)
check(
    "transfer_complete_reference_match",
    len(match) == len(new_transfer[new_transfer.predictor == "ml"]) == len(old_transfer),
)
check("transfer_MAE_matches_original", near(match.MAE_new, match.MAE_old))
check("transfer_accuracy_matches_original", near(match.accuracy_new, match.accuracy_old))
cohorts = read_csv(ROOT / "results/sensitivity_cohorts.csv")
future = [
    f
    for f in json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
    if f["repeat"] == 0
]
overlap = read_csv(W34 / "verification/feature_overlap_pairs.csv")
for f in future:
    drop = set(
        overlap.loc[(overlap.task == f["task"]) & (overlap.alloy == f["alloy"]), "test_row_id"]
    )
    row = cohorts[(cohorts.task == f["task"]) & (cohorts.alloy == f["alloy"])].iloc[0]
    check(
        "sensitivity_counts_include_zero_survivors",
        row.retained_test_rows == len(set(f["test_row_ids"]) - drop)
        and row.original_test_rows == 20,
    )
figures = json.loads((ROOT / "figures/index.json").read_text())
for fig in figures:
    check(
        "figure_all_export_formats",
        all(
            (ROOT / fig[fmt]).is_file() and (ROOT / fig[fmt]).stat().st_size > 1000
            for fmt in ["png", "svg", "pdf"]
        ),
    )
    check(
        "figure_source_tables_exist",
        all((ROOT / "results" / table).is_file() for table in fig["source_tables"]),
    )
failures = [x for x in checks if not x["passed"]]
result = {"passed": len(checks) - len(failures), "failed": len(failures), "failures": failures}
write_json(ROOT / "verification/diagnostic_checks.json", result)
print(json.dumps(result, indent=2))
if failures:
    raise SystemExit(1)
