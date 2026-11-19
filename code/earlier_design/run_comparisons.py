"""New simple-predictor comparisons, preserving all earlier experiments."""
import json
import numpy as np
import pandas as pd
from common import *


def main():
    protocol = json.loads((ROOT/"protocol/supplementary_spec.json").read_text())
    for rel, expected in protocol["dependencies"].items():
        assert sha(BASE/rel) == expected, f"Frozen dependency changed: {rel}"
    spec = json.loads((W12/"protocol/frozen.json").read_text())
    splits = json.loads((W12/"protocol/splits.json").read_text())
    future = json.loads((W12/"protocol/future_calibration_splits.json").read_text())
    data = {t: read_csv(W12/f"data/{t}.csv", dtype={"source": str}).set_index("row_id") for t in spec["tasks"]}
    ml = read_csv(W34/"results/update_predictions.csv", dtype={"source": str})
    ml["predictor"] = "ml"
    simple, thresholds, scores_out, fitted = [], [], [], []
    for i, split in enumerate(splits):
        selected = [f for f in future if f["task"] == split["task"] and f["alloy"] == split["alloy"]]
        if split["protocol"] != "leave_one_alloy_out" or not selected:
            continue
        task = split["task"]; cls = task == "classification"
        d = data[task]; target = spec["tasks"][task]["target"]
        train = d.loc[split["train"], target]
        key = f"{task}_{split['protocol']}_{i:03d}_s{split['seed']}"
        meta = dict(key=key, task=task, alloy=split["alloy"], model_seed=split["seed"], predictor="simple")
        if cls:
            probabilities = np.array([(train == label).mean() for label in LABELS])
            point = str(train.mode().iloc[0])
        else:
            point = float(train.mean())
        fitted.append(meta | {"n_train": len(train), "point": point,
                              "probabilities_json": json.dumps(dict(zip(LABELS, probabilities))) if cls else ""})

        def scores(ids):
            truth = d.loc[ids, target]
            return np.array([1-probabilities[LABELS.index(v)] for v in truth]) if cls else abs(truth.to_numpy(float)-point)

        source_scores = scores(split["calibration"])
        for f in selected:
            cohort = "primary_paired" if f["primary_budget_comparison"] else "secondary_9_only"
            for budget in [0, 9, 20]:
                ids = split["calibration"] if budget == 0 else f[f"calibration{budget}"]
                if ids is None:
                    continue
                assert set(ids).isdisjoint(f["test_row_ids"])
                values = source_scores if budget == 0 else scores(ids)
                k, q = quantile(values)
                context = meta | dict(cohort=cohort, repeat=f["repeat"], budget=budget, n_cal=len(ids), k=k, q=q)
                thresholds.append(context)
                for rid, value in zip(ids, values):
                    # Source calibration needs only one stored copy per model.
                    if budget or f["repeat"] == 0:
                        scores_out.append(context | {"row_id": rid, "score": float(value)})
                test = d.loc[f["test_row_ids"]]
                out = pd.DataFrame({"row_id": test.index, "source": test.source.to_numpy(),
                                    "actual": test[target].to_numpy(), "predicted": point})
                for name, value in context.items():
                    out[name] = value
                if cls:
                    catset = [label for label, p in zip(LABELS, probabilities) if 1-p <= q]
                    for label, p in zip(LABELS, probabilities):
                        out["prob_"+label] = p
                    out["set_json"] = json.dumps(catset)
                    out["set_size"] = len(catset)
                    out["covered"] = out.actual.isin(catset)
                else:
                    out["lower"], out["upper"] = point-q, point+q
                    out["width"] = 2*q
                    out["covered"] = (out.actual >= point-q) & (out.actual <= point+q)
                    out["interval_score"] = 2*q + 20*np.maximum(point-q-out.actual, 0) + 20*np.maximum(out.actual-point-q, 0)
                simple.append(out)
        print(key, "complete", flush=True)
    combined = pd.concat([ml, *simple], ignore_index=True)
    combined.to_csv(ROOT/"results/comparison_predictions.csv", index=False)
    pd.DataFrame(fitted).to_csv(ROOT/"results/simple_fitted_constants.csv", index=False)
    pd.DataFrame(thresholds).to_csv(ROOT/"results/simple_thresholds.csv", index=False)
    pd.DataFrame(scores_out).to_csv(ROOT/"results/simple_calibration_scores.csv", index=False)
    _, alloy, summary = aggregate(metric_table(combined))
    keys = ["task", "cohort", "alloy", "budget"]
    values = [m for m in METRICS if m in alloy]
    paired = alloy[alloy.predictor == "ml"].merge(alloy[alloy.predictor == "simple"], on=keys, suffixes=("_ml", "_simple"))
    for v in values:
        paired[v+"_ml_minus_simple"] = paired[v+"_ml"] - paired[v+"_simple"]
    paired.to_csv(ROOT/"results/paired_differences.csv", index=False)
    # Independent cohorts and equal-alloy weighting also apply to the reduced test sets.
    overlap = read_csv(W34/"verification/feature_overlap_pairs.csv")
    excluded = set(zip(overlap.task, overlap.alloy, overlap.test_row_id))
    keep = [key not in excluded for key in zip(combined.task, combined.alloy, combined.row_id)]
    reduced = combined.loc[keep]
    aggregate(metric_table(reduced), prefix="sensitivity_")
    counts = reduced.groupby(["task", "cohort", "alloy"]).row_id.nunique().rename("retained_test_rows").reset_index()
    counts["original_test_rows"] = 20
    counts.to_csv(ROOT/"results/sensitivity_cohorts.csv", index=False)
    print(summary[summary.cohort == "primary_paired"][["task", "budget", "predictor", "n_alloys", "coverage", "mean_width", "mean_interval_score", "mean_set_size"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
