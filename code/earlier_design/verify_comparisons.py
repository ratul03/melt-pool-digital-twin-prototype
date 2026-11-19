"""Independent check of new calculations, partitions and all summary levels."""

from pathlib import Path
import json
import math
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
W12, W34 = BASE / "Thesis_C_Weeks_1_2", BASE / "Thesis_C_Weeks_3_4"
checks = []
boundary_rows = []


def check(name, passed, context=""):
    checks.append({"check": name, "passed": bool(passed), "context": str(context)})


def csv(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def near(a, b):
    return np.allclose(
        np.asarray(a, float), np.asarray(b, float), rtol=1e-10, atol=1e-8, equal_nan=True
    )


def main():
    config = json.loads((ROOT / "protocol/supplementary_spec.json").read_text())
    for rel, expected in config["dependencies"].items():
        check(
            "inherited_artifact_unchanged",
            hashlib.sha256((BASE / rel).read_bytes()).hexdigest() == expected,
            rel,
        )
    spec = json.loads((W12 / "protocol/frozen.json").read_text())
    splits = json.loads((W12 / "protocol/splits.json").read_text())
    splitmap = {
        f"{s['task']}_{s['protocol']}_{i:03d}_s{s['seed']}": s for i, s in enumerate(splits)
    }
    future = {
        (f["task"], f["alloy"], f["repeat"]): f
        for f in json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
    }
    data = {t: csv(W12 / f"data/{t}.csv").set_index("row_id") for t in spec["tasks"]}
    labels = ["LOF", "balling", "desirable", "keyhole"]
    p = csv(ROOT / "results/comparison_predictions.csv")
    source = csv(W34 / "results/update_predictions.csv")
    keys = ["key", "repeat", "budget", "row_id"]
    old = source.sort_values(keys).reset_index(drop=True)
    retained = p[p.predictor == "ml"].sort_values(keys).reset_index(drop=True)
    check("all_original_prediction_rows_retained", len(old) == len(retained))
    for col in old:
        if col in ["actual", "predicted"]:
            cls = old.task == "classification"
            check(
                "original_class_values_unchanged",
                np.array_equal(old.loc[cls, col].astype(str), retained.loc[cls, col].astype(str)),
                col,
            )
            check(
                "original_numeric_values_unchanged",
                near(old.loc[~cls, col], retained.loc[~cls, col]),
                col,
            )
        elif pd.api.types.is_numeric_dtype(old[col]):
            check("original_numeric_column_unchanged", near(old[col], retained[col]), col)
        else:
            check(
                "original_column_unchanged",
                old[col].fillna("").astype(str).equals(retained[col].fillna("").astype(str)),
                col,
            )
    tables = csv(ROOT / "results/conditions.csv")
    condkeys = ["task", "cohort", "alloy", "model_seed", "repeat", "budget", "predictor"]
    tables = tables.set_index(condkeys)
    computed = []
    scoretable = csv(ROOT / "results/simple_calibration_scores.csv")
    storedscores = {
        (k, int(r), int(b)): g for (k, r, b), g in scoretable.groupby(["key", "repeat", "budget"])
    }
    for key, g in p.groupby(condkeys, sort=False):
        meta = dict(zip(condkeys, key))
        split = splitmap[g.key.iloc[0]]
        f = future[(meta["task"], meta["alloy"], meta["repeat"])]
        d = data[meta["task"]]
        target = spec["tasks"][meta["task"]]["target"]
        check(
            "identical_final_test_records",
            set(g.row_id) == set(f["test_row_ids"]) and len(g) == 20,
            key,
        )
        check(
            "fitting_calibration_test_disjoint",
            not (
                set(split["train"]) & set(split["calibration"])
                or set(split["train"]) & set(split["test"])
                or set(split["calibration"]) & set(split["test"])
            ),
            key,
        )
        check(
            "no_target_alloy_in_source_roles",
            meta["alloy"] not in set(d.loc[split["train"] + split["calibration"], "alloy"]),
            key,
        )
        check(
            "test_excluded_from_measurement_pool",
            not set(f["test_row_ids"]) & set(f["pool_row_ids"]),
            key,
        )
        check(
            "nested_measurements",
            f["calibration20"] is None or set(f["calibration9"]) <= set(f["calibration20"]),
            key,
        )
        cls = meta["task"] == "classification"
        actual = d.loc[g.row_id, target].to_numpy()
        check(
            "actual_matches_frozen_data",
            np.array_equal(actual, g.actual.to_numpy()) if cls else near(actual, g.actual),
            key,
        )
        if meta["predictor"] == "simple":
            ytrain = d.loc[split["train"], target]
            ids = split["calibration"] if meta["budget"] == 0 else f[f"calibration{meta['budget']}"]
            check(
                "calibration_count_and_roles",
                len(ids) == (len(split["calibration"]) if meta["budget"] == 0 else meta["budget"])
                and not set(ids) & set(g.row_id),
                key,
            )
            if cls:
                freq = {
                    lab: np.count_nonzero(ytrain.to_numpy() == lab) / len(ytrain) for lab in labels
                }
                point = sorted(freq, key=lambda lab: (-freq[lab], lab))[0]
                check(
                    "training_only_class_frequencies",
                    all(near(g["prob_" + lab], freq[lab]) for lab in labels),
                    key,
                )
                check("training_only_majority", g.predicted.eq(point).all(), key)
                scores = [1 - freq[y] for y in d.loc[ids, target]]
            else:
                point = sum(float(v) for v in ytrain) / len(ytrain)
                check("training_only_mean", near(g.predicted, point), key)
                scores = [abs(float(y) - point) for y in d.loc[ids, target]]
            rank = math.ceil(9 * (len(scores) + 1) / 10)
            q = sorted(scores)[rank - 1]
            check("finite_sample_rank", near(g.k, rank) and near(g.q, q), key)
            sc = storedscores[
                (g.key.iloc[0], 0 if meta["budget"] == 0 else meta["repeat"], meta["budget"])
            ]
            check(
                "stored_calibration_scores",
                sc.row_id.tolist() == ids and near(sc.score, scores),
                key,
            )
        metrics = {}
        if cls:
            probs = g[["prob_" + lab for lab in labels]].to_numpy(float)
            membership = np.array([[lab in json.loads(s) for lab in labels] for s in g.set_json])
            rule = 1 - probs <= g.q.to_numpy()[:, None]
            check("set_inclusion_rule", np.array_equal(rule, membership), key)
            sizes = membership.sum(1)
            cov = membership[np.arange(len(g)), [labels.index(x) for x in actual]]
            f1 = []
            for lab in labels:
                tp = ((g.actual == lab) & (g.predicted == lab)).sum()
                denom = (g.actual == lab).sum() + (g.predicted == lab).sum()
                f1.append(2 * tp / denom if denom else 0)
            metrics.update(
                coverage=cov.mean(),
                accuracy=(g.actual == g.predicted).mean(),
                macro_f1=np.mean(f1),
                mean_set_size=sizes.mean(),
                empty_set_fraction=(sizes == 0).mean(),
                singleton_fraction=(sizes == 1).mean(),
                full_set_fraction=(sizes == 4).mean(),
            )
        else:
            y, pred = g.actual.to_numpy(float), g.predicted.to_numpy(float)
            lo, hi = pred - g.q.to_numpy(), pred + g.q.to_numpy()
            recomputed_cov = (y >= lo) & (y <= hi)
            cov = g.covered.to_numpy(bool)
            boundary = (abs(y - lo) <= 1e-8) | (abs(y - hi) <= 1e-8)
            check(
                "coverage_from_bounds_except_numerical_ties",
                np.all((cov == recomputed_cov) | boundary),
                key,
            )
            for position in np.flatnonzero(cov != recomputed_cov):
                boundary_rows.append(
                    meta
                    | {
                        "row_id": int(g.row_id.iloc[position]),
                        "actual": float(y[position]),
                        "recomputed_lower": float(lo[position]),
                        "recomputed_upper": float(hi[position]),
                        "stored_covered": bool(cov[position]),
                        "endpoint_distance_um": float(
                            min(abs(y[position] - lo[position]), abs(y[position] - hi[position]))
                        ),
                    }
                )
            lower = 20 * np.maximum(lo - y, 0)
            upper = 20 * np.maximum(y - hi, 0)
            score = hi - lo + lower + upper
            check(
                "bounds_and_score",
                near(g.lower, lo) and near(g.upper, hi) and near(g.interval_score, score),
                key,
            )
            metrics.update(
                coverage=cov.mean(),
                MAE=abs(y - pred).mean(),
                RMSE=np.sqrt(((y - pred) ** 2).mean()),
                R2=(
                    1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                    if np.var(y)
                    else np.nan
                ),
                mean_width=(hi - lo).mean(),
                mean_interval_score=score.mean(),
                lower_miss_penalty=lower.mean(),
                upper_miss_penalty=upper.mean(),
                negative_point_fraction=(pred < 0).mean(),
                negative_lower_fraction=(lo < 0).mean(),
            )
        check("row_coverage", np.array_equal(g.covered.to_numpy(bool), cov), key)
        check(
            "condition_metrics",
            all(near(tables.loc[key, col], value) for col, value in metrics.items()),
            key,
        )
        computed.append(meta | metrics)
    calculated = pd.DataFrame(computed)
    metrics = [c for c in calculated if c not in condkeys]
    stage = calculated.groupby([c for c in condkeys if c != "repeat"])[metrics].mean().reset_index()
    alloy = (
        stage.groupby([c for c in condkeys if c not in {"repeat", "model_seed"}])[metrics]
        .mean()
        .reset_index()
    )
    summarykeys = ["task", "cohort", "budget", "predictor"]
    summary = alloy.groupby(summarykeys)[metrics].mean()
    stored = csv(ROOT / "results/summary.csv").set_index(summarykeys)
    for col in metrics:
        check("equal_alloy_aggregation", near(stored.loc[summary.index, col], summary[col]), col)
    check("paired_predictor_row_counts", p.groupby("predictor").size().nunique() == 1)
    for key, z in calculated.groupby(
        ["task", "cohort", "alloy", "model_seed", "repeat", "predictor"]
    ):
        metric = "accuracy" if key[0] == "classification" else "MAE"
        check("updates_leave_point_predictions_fixed", near(z[metric], z[metric].iloc[0]), key)
    # Recompute the reduced-cohort metrics without importing experiment code.
    overlap = csv(W34 / "verification/feature_overlap_pairs.csv")
    excluded = set(zip(overlap.task, overlap.alloy, overlap.test_row_id))
    reduced = p[[key not in excluded for key in zip(p.task, p.alloy, p.row_id)]]
    saved_sens = csv(ROOT / "results/sensitivity_conditions.csv").set_index(condkeys)
    for key, g in reduced.groupby(condkeys):
        cov = g.covered.mean()
        check(
            "sensitivity_counts_and_coverage",
            near(saved_sens.loc[key, "n_test"], len(g))
            and near(saved_sens.loc[key, "coverage"], cov),
            key,
        )
        if key[0] != "classification":
            check(
                "sensitivity_interval_score",
                near(saved_sens.loc[key, "mean_interval_score"], g.interval_score.mean()),
                key,
            )
    fail = [c for c in checks if not c["passed"]]
    pd.DataFrame(boundary_rows).to_csv(
        ROOT / "verification/numerical_boundary_ties.csv", index=False
    )
    result = {
        "passed": len(checks) - len(fail),
        "failed": len(fail),
        "failures": fail,
        "checks_by_type": pd.Series([c["check"] for c in checks]).value_counts().to_dict(),
        "prediction_rows": len(p),
        "conditions": len(tables),
        "numerical_boundary_rows": len(boundary_rows),
        "boundary_policy": "Keep original evaluated coverage booleans. Recomputed bounds may differ at endpoint ties after CSV rounding; every discrepancy must be within 1e-8 um. Bounds, widths and scores are independently checked.",
        "scope": "New comparison arithmetic, label roles, inherited-output identity and aggregation.",
    }
    (ROOT / "verification/comparison_checks.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
