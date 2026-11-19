"""Reproduce fixed Week 2 predictors, then run the frozen Week 3 calibration study.

Run from the existing pinned modelling environment. Writes only to Weeks_3_4.
No fitting, selection, clipping, or tuning uses the fixed final test observations.
"""

from pathlib import Path
import sys
import json
import hashlib
import warnings
import time
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT.parent / "Thesis_C_Weeks_1_2"
LABELS = ["LOF", "balling", "desirable", "keyhole"]
sys.path.insert(0, str(PREV / "inputs/code"))
import reproduce as inherited


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def threshold(scores):
    """Exact integer ceil(0.9*(n+1)); n=9 -> rank9, n=20 -> rank19."""
    vals = np.asarray(scores, dtype=float)
    if not len(vals) or not np.isfinite(vals).all():
        raise ValueError("Calibration scores must be nonempty and finite")
    k = (9 * (len(vals) + 1) + 9) // 10
    return k, float(np.sort(vals)[k - 1]) if k <= len(vals) else float("inf")


def predict(art, d):
    x = art["scaler"].transform(d[art["features"]].to_numpy())
    p = art["model"].predict(x)
    if art["target_scaler"] is not None:
        p = art["target_scaler"].inverse_transform(p.reshape(-1, 1)).ravel()
    frame = pd.DataFrame(
        {
            "row_id": d.index,
            "alloy": d.alloy.to_numpy(),
            "source": d.source.to_numpy(),
            "actual": d[art["target"]].to_numpy(),
            "predicted": p,
        }
    )
    if art["target"] == "label":
        probs = art["model"].predict_proba(x)
        for lab in LABELS:
            frame["prob_" + lab] = (
                probs[:, list(art["model"].classes_).index(lab)]
                if lab in art["model"].classes_
                else 0.0
            )
        prob_array = frame[["prob_" + lab for lab in LABELS]].to_numpy()
        frame["score"] = (
            1 - prob_array[np.arange(len(frame)), [LABELS.index(x) for x in frame.actual]]
        )
    else:
        frame["score"] = np.abs(frame.actual - frame.predicted)
    return frame


def evaluate(frame, q, classification):
    out = frame.copy()
    if classification:
        included = 1 - out[["prob_" + lab for lab in LABELS]].to_numpy() <= q
        out["set_json"] = [
            json.dumps([lab for lab, keep in zip(LABELS, r) if keep]) for r in included
        ]
        out["set_size"] = included.sum(axis=1)
        out["covered"] = included[np.arange(len(out)), [LABELS.index(x) for x in out.actual]]
    else:
        out["lower"] = out.predicted - q
        out["upper"] = out.predicted + q
        out["covered"] = (out.actual >= out.lower) & (out.actual <= out.upper)
        out["width"] = out.upper - out.lower
        out["interval_score"] = (
            out.width
            + 20 * np.maximum(out.lower - out.actual, 0)
            + 20 * np.maximum(out.actual - out.upper, 0)
        )
    return out


def metric_row(frame, cls):
    r = {"n_test": len(frame), "coverage": float(frame.covered.mean())}
    if cls:
        r.update(
            mean_set_size=float(frame.set_size.mean()),
            empty_set_fraction=float(frame.set_size.eq(0).mean()),
            singleton_fraction=float(frame.set_size.eq(1).mean()),
            full_set_fraction=float(frame.set_size.eq(4).mean()),
            accuracy=float(frame.actual.eq(frame.predicted).mean()),
        )
    else:
        r.update(
            mean_width=float(frame.width.mean()),
            mean_interval_score=float(frame.interval_score.mean()),
            MAE=float(np.abs(frame.actual - frame.predicted).mean()),
            negative_point_count=int(frame.predicted.lt(0).sum()),
            negative_lower_count=int(frame.lower.lt(0).sum()),
            negative_point_fraction=float(frame.predicted.lt(0).mean()),
            negative_lower_fraction=float(frame.lower.lt(0).mean()),
        )
    return r


def annotate(frame, meta):
    frame = frame.copy()
    for k, v in meta.items():
        # Retain actual per-row alloy for random baselines; heldout_alloy is in meta.
        if k == "alloy" and "alloy" in frame:
            continue
        frame[k] = v
    return frame


def main():
    for folder in ["protocol", "results/runs", "models", "verification"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    spec = json.loads((PREV / "protocol/frozen.json").read_text())
    splits = json.loads((PREV / "protocol/splits.json").read_text())
    future = json.loads((PREV / "protocol/future_calibration_splits.json").read_text())
    data = {
        t: pd.read_csv(PREV / "data" / f"{t}.csv", dtype={"source": str}).set_index("row_id")
        for t in spec["tasks"]
    }
    for rel, h in spec["input_hashes"].items():
        assert digest(PREV / rel) == h, rel
    source_manifest = {
        str(p.relative_to(PREV)).replace("\\", "/"): digest(p)
        for p in sorted(PREV.rglob("*"))
        if p.is_file()
    }
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "saved_before_uncertainty_results",
        "source_package": PREV.name,
        "source_manifest": source_manifest,
        "seeds": [0, 1, 2, 3, 4],
        "repeats": 20,
        "nominal_coverage": 0.9,
        "primary_outcome": "depth",
        "primary_contrast": "target20 minus unchanged source calibration, equal-alloy mean interval score",
        "supporting_contrast": "target9 and target20 measurement budgets",
        "secondary_outputs": ["width", "length", "classification"],
        "cohorts": {
            "primary_paired": "same frozen alloys eligible for 9 and20, at least40 total rows",
            "secondary_9_only": "frozen9-only alloys, at least29 but fewer40 rows",
        },
        "geometry_score": "abs(actual-predicted)",
        "classification_score": "1-p_true",
        "rank": "ceil(9*(n+1)/10) computed with integer arithmetic; q=inf if k>n",
        "geometry_intervals": "predicted plus/minus q; inclusive; no clipping",
        "classification_sets": "include each fixed label if1-p_label<=q; absent fitted class p0; no forced nonempty",
        "fixed_labels": LABELS,
        "interval_score": "width+20*lower_miss_distance+20*upper_miss_distance",
        "update": "replace source threshold with target-only threshold; point model unchanged",
        "reporting": "mean over selection repeats within model seed, then seeds within alloy, then equal-alloy mean; variation is descriptive, not independent samples or confidence intervals",
        "source_baseline": "original random and whole-alloy held-out Week2 rows; separate from fixed20 final-test update cohort",
        "guarantee": "nominal marginal conformal coverage relies on calibration/test exchangeability; no source-to-target guarantee under distribution shift; correlated records and source confounding limit interpretation",
        "test_reuse": "Week2 outcomes already reported; future partitions were frozen before Week2 scoring and no new settings tuned on test labels",
        "model_reconstruction": "load saved seed0; deterministic refit seeds1-4 with inherited train partitions/settings, confirm original test predictions rtol1e-12 atol1e-9 and classifier probabilities atol1e-12",
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "script_sha256_at_launch": digest(__file__),
    }
    frozen = ROOT / "protocol/calibration_spec.json"
    if not frozen.exists():
        write_json(frozen, protocol)
    else:
        old = json.loads(frozen.read_text())
        assert (
            old["source_manifest"] == source_manifest
        ), "Source package changed after protocol was frozen"
    future_map = {}
    selections = []
    for f in future:
        future_map.setdefault((f["task"], f["alloy"]), []).append(f)
        for budget in [9, 20]:
            ids = f["calibration" + str(budget)]
            if ids is not None:
                selections.append(
                    {
                        "task": f["task"],
                        "alloy": f["alloy"],
                        "repeat": f["repeat"],
                        "budget": budget,
                        "cohort": (
                            "primary_paired"
                            if f["primary_budget_comparison"]
                            else "secondary_9_only"
                        ),
                        "pool_n": len(f["pool_row_ids"]),
                        "test_row_ids_json": json.dumps(f["test_row_ids"]),
                        "calibration_row_ids_json": json.dumps(ids),
                        "selection_set_sha256": hashlib.sha256(
                            json.dumps(sorted(ids)).encode()
                        ).hexdigest(),
                    }
                )
    sel = pd.DataFrame(selections)
    sel.to_csv(ROOT / "protocol/target_selections.csv", index=False)
    uniq = (
        sel.groupby(["task", "alloy", "budget", "cohort"], sort=False)
        .agg(
            repeats=("repeat", "size"),
            unique_calibration_sets=("selection_set_sha256", "nunique"),
            pool_n=("pool_n", "first"),
        )
        .reset_index()
    )
    uniq.to_csv(ROOT / "results/selection_diversity.csv", index=False)
    with threadpool_limits(limits=1):
        for i, s in enumerate(splits):
            key = f"{s['task']}_{s['protocol']}_{i:03d}_s{s['seed']}"
            done = ROOT / "results/runs" / f"{key}_done.json"
            if done.exists():
                continue
            t = s["task"]
            cls = t == "classification"
            cfg = spec["tasks"][t]
            d = data[t]
            start = time.time()
            tr, ca, te = [d.loc[s[x]] for x in ["train", "calibration", "test"]]
            assert not (
                set(tr.index) & set(ca.index)
                or set(tr.index) & set(te.index)
                or set(ca.index) & set(te.index)
            )
            model_path = ROOT / "models" / f"{key}.joblib"
            warns = []
            if s["seed"] == 0:
                original = json.loads((PREV / "results/runs" / f"{key}.json").read_text())
                art = joblib.load(PREV / original["saved_model"])
                origin = "loaded_original_seed0"
            elif model_path.exists():
                art = joblib.load(model_path)
                origin = "loaded_verified_refit"
            else:
                scaler = StandardScaler().fit(tr[cfg["features"]].to_numpy())
                target_scaler = None
                y = tr[cfg["target"]].to_numpy()
                if cfg["target_scaling"]:
                    target_scaler = StandardScaler().fit(y.reshape(-1, 1))
                    y = target_scaler.transform(y.reshape(-1, 1)).ravel()
                model = inherited.estimator(cfg["model"], t, s["seed"], "corrected")
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    model.fit(scaler.transform(tr[cfg["features"]].to_numpy()), y)
                warns = [str(w.message) for w in ws]
                art = {
                    "model": model,
                    "scaler": scaler,
                    "target_scaler": target_scaler,
                    "features": cfg["features"],
                    "target": cfg["target"],
                    "split": s,
                    "protocol_sha256": digest(PREV / "protocol/frozen.json"),
                    "research_only": True,
                }
                origin = "deterministic_refit"
            pred = predict(art, te)
            prior = pd.read_csv(PREV / "results/runs" / f"{key}_predictions.csv")
            assert prior.row_id.tolist() == pred.row_id.tolist()
            if cls:
                assert np.array_equal(pred.predicted, prior.predicted), key
                err = float(
                    np.max(
                        np.abs(
                            pred[["prob_" + x for x in LABELS]].to_numpy()
                            - prior[["prob_" + x for x in LABELS]].to_numpy()
                        )
                    )
                )
                assert err <= 1e-12, (key, err)
            else:
                err = float(np.max(np.abs(pred.predicted - prior.predicted)))
                assert np.allclose(pred.predicted, prior.predicted, rtol=1e-12, atol=1e-9), (
                    key,
                    err,
                )
            if s["seed"] != 0 and not model_path.exists():
                joblib.dump(art, model_path, compress=3)
            common = {
                "key": key,
                "task": t,
                "protocol": s["protocol"],
                "alloy": s["alloy"],
                "heldout_alloy": s["alloy"],
                "model_seed": s["seed"],
            }
            source_scores = predict(art, ca)
            k, q = threshold(source_scores.score)
            source_meta = common | {
                "repeat": -1,
                "budget": 0,
                "n_cal": len(ca),
                "k": k,
                "q": q,
                "cohort": "week2_original_test",
                "calibration_phase": "source",
            }
            source_cal = annotate(source_scores, source_meta)
            initial = annotate(evaluate(pred, q, cls), source_meta)
            cal_frames = [source_cal]
            threshold_rows = [source_meta]
            updates = []
            metrics = []
            for f in (
                future_map.get((t, s["alloy"]), [])
                if s["protocol"] == "leave_one_alloy_out"
                else []
            ):
                final = d.loc[f["test_row_ids"]]
                assert len(final) == 20 and not set(final.index) & (
                    set(tr.index) | set(ca.index) | set(f["pool_row_ids"])
                )
                cohort = "primary_paired" if f["primary_budget_comparison"] else "secondary_9_only"
                final_pred = pred.set_index("row_id").loc[f["test_row_ids"]].reset_index()
                assert final_pred.alloy.eq(s["alloy"]).all()
                for budget in [0, 9, 20]:
                    ids = f.get("calibration" + str(budget)) if budget else s["calibration"]
                    if ids is None:
                        continue
                    if budget:
                        assert len(set(ids)) == budget and set(ids) <= set(f["pool_row_ids"])
                        scores = pred.set_index("row_id").loc[ids].reset_index()
                        kk, qq = threshold(scores.score)
                    else:
                        scores = source_scores
                        kk, qq = k, q
                    meta = common | {
                        "repeat": f["repeat"],
                        "budget": budget,
                        "n_cal": len(scores),
                        "k": kk,
                        "q": qq,
                        "cohort": cohort,
                        "calibration_phase": "target" if budget else "source",
                    }
                    if budget:
                        cal_frames.append(annotate(scores, meta))
                    threshold_rows.append(meta)
                    result = annotate(evaluate(final_pred, qq, cls), meta)
                    updates.append(result)
                    metrics.append(meta | metric_row(result, cls))
            prefix = ROOT / "results/runs" / key
            source_cal_path = str(prefix) + "_calibration.csv"
            pd.concat(cal_frames, ignore_index=True).to_csv(source_cal_path, index=False)
            initial.to_csv(str(prefix) + "_initial.csv", index=False)
            pd.DataFrame(threshold_rows).to_csv(str(prefix) + "_thresholds.csv", index=False)
            if updates:
                pd.concat(updates, ignore_index=True).to_csv(
                    str(prefix) + "_updates.csv", index=False
                )
                pd.DataFrame(metrics).to_csv(str(prefix) + "_metrics.csv", index=False)
            write_json(
                done,
                {
                    "key": key,
                    "task": t,
                    "seed": s["seed"],
                    "origin": origin,
                    "old_predictions_verified": True,
                    "max_abs_difference": err,
                    "n_train": len(tr),
                    "n_source_cal": len(ca),
                    "n_old_test": len(te),
                    "warnings": warns,
                    "seconds": time.time() - start,
                    "artifact": (
                        str(model_path.relative_to(ROOT)) if s["seed"] else original["saved_model"]
                    ),
                },
            )
            print(
                i + 1,
                "/",
                len(splits),
                key,
                origin,
                "prediction_difference",
                err,
                "update_conditions",
                len(metrics),
                "seconds",
                round(time.time() - start, 2),
                flush=True,
            )
    aggregate(data, splits, spec, source_manifest)


def aggregate(data, splits, spec, source_manifest):
    runs = ROOT / "results/runs"
    tables = {}
    for suffix, dest in [
        ("calibration", "calibration_scores"),
        ("initial", "baseline_initial_predictions"),
        ("thresholds", "thresholds"),
        ("updates", "update_predictions"),
        ("metrics", "per_condition_metrics"),
    ]:
        frames = [
            pd.read_csv(p, dtype={"source": str}, keep_default_na=True)
            for p in sorted(runs.glob("*_" + suffix + ".csv"))
        ]
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(ROOT / "results" / (dest + ".csv"), index=False)
        tables[dest] = frame
    runlog = [json.loads(p.read_text()) for p in sorted(runs.glob("*_done.json"))]
    pd.DataFrame(runlog).to_csv(ROOT / "results/model_reproduction_log.csv", index=False)
    assert len(runlog) == len(splits)
    met = tables["per_condition_metrics"]
    numeric = [
        "coverage",
        "mean_width",
        "mean_interval_score",
        "MAE",
        "negative_point_fraction",
        "negative_lower_fraction",
        "mean_set_size",
        "empty_set_fraction",
        "singleton_fraction",
        "full_set_fraction",
        "accuracy",
    ]
    alloy_seed = (
        met.groupby(["task", "cohort", "alloy", "model_seed", "budget"], sort=False)[numeric]
        .mean()
        .reset_index()
    )
    alloy_seed.to_csv(ROOT / "results/per_alloy_seed_summary.csv", index=False)
    alloy = (
        alloy_seed.groupby(["task", "cohort", "alloy", "budget"], sort=False)[numeric]
        .mean()
        .reset_index()
    )
    alloy.to_csv(ROOT / "results/per_alloy_summary.csv", index=False)
    summary = alloy.groupby(["task", "cohort", "budget"], sort=False)[numeric].mean().reset_index()
    counts = (
        alloy.groupby(["task", "cohort", "budget"], sort=False)
        .alloy.nunique()
        .rename("n_alloys")
        .reset_index()
    )
    summary = summary.merge(counts, on=["task", "cohort", "budget"])
    summary["unique_test_rows"] = summary.n_alloys * 20
    summary.to_csv(ROOT / "results/summary.csv", index=False)
    contrasts = []
    for (task, cohort, alloy_name), group in alloy.groupby(["task", "cohort", "alloy"], sort=False):
        base = group[group.budget.eq(0)].iloc[0]
        for _, row in group[group.budget.ne(0)].iterrows():
            contrasts.append(
                {
                    "task": task,
                    "cohort": cohort,
                    "alloy": alloy_name,
                    "budget": row.budget,
                    **{"delta_" + x: row[x] - base[x] for x in numeric},
                }
            )
    delta = pd.DataFrame(contrasts)
    delta.to_csv(ROOT / "results/per_alloy_update_deltas.csv", index=False)
    delta.groupby(["task", "cohort", "budget"], sort=False)[
        ["delta_" + x for x in numeric]
    ].mean().reset_index().to_csv(ROOT / "results/update_deltas.csv", index=False)
    # Baseline coverage gap: use only Week2 primary alloys (>=20 rows), same cohort definitions,
    # retain row-weighted and equal-alloy summaries separately and never blend with fixed20 tests.
    baseline = tables["baseline_initial_predictions"]
    initial_metrics = []
    counts = pd.read_csv(PREV / "data/alloy_counts.csv")
    for (task, protocol, seed), group in baseline.groupby(
        ["task", "protocol", "model_seed"], sort=False
    ):
        primary = counts.loc[counts.task.eq(task) & counts.primary, "alloy"].tolist()
        group = group[group.alloy.isin(primary)]
        initial_metrics.append(
            {
                "task": task,
                "protocol": protocol,
                "model_seed": seed,
                "weighting": "row_weighted",
                "alloy": "__pooled__",
                "n_alloys": group.alloy.nunique(),
                **metric_row(group, task == "classification"),
            }
        )
        for alloy_name, ag in group.groupby("alloy", sort=False):
            initial_metrics.append(
                {
                    "task": task,
                    "protocol": protocol,
                    "model_seed": seed,
                    "weighting": "per_alloy",
                    "alloy": alloy_name,
                    "n_alloys": 1,
                    **metric_row(ag, task == "classification"),
                }
            )
    initial_m = pd.DataFrame(initial_metrics)
    initial_m.to_csv(ROOT / "results/baseline_initial_metrics.csv", index=False)
    for weighting in ["row_weighted", "per_alloy"]:
        subset = initial_m[initial_m.weighting.eq(weighting)]
        per_seed = (
            subset.groupby(["task", "protocol", "model_seed"], sort=False)[numeric]
            .mean()
            .reset_index()
        )
        grouped = per_seed.groupby(["task", "protocol"], sort=False)[numeric].agg(["mean", "std"])
        grouped.columns = ["_".join(x) for x in grouped.columns]
        grouped.reset_index().to_csv(
            ROOT / "results" / f"baseline_initial_{weighting}_summary.csv", index=False
        )
    # Validation summary comprises procedural checks; independent reviewer checks numerical outputs separately.
    source_unchanged = all(digest(PREV / rel) == h for rel, h in source_manifest.items())
    assert source_unchanged
    update = tables["update_predictions"]
    key_cols = ["task", "alloy", "model_seed", "repeat", "row_id"]
    assert update.groupby(key_cols).predicted.nunique().eq(1).all()
    for c in ["prob_" + x for x in LABELS]:
        assert update[update.task.eq("classification")].groupby(key_cols)[c].nunique().eq(1).all()
    assert tables["thresholds"].q.notna().all()
    checks = {
        "all_original_files_unchanged": source_unchanged,
        "all_250_original_prediction_vectors_recovered": len(runlog) == 250,
        "max_original_prediction_difference": max(x["max_abs_difference"] for x in runlog),
        "fixed_point_predictions_across_updates": True,
        "fixed_class_probabilities_across_updates": True,
        "source_input_hashes_checked": True,
        "original_partitions_disjoint": True,
        "target_pool_and_final_test_disjoint": True,
        "n_update_conditions": len(met),
        "n_update_prediction_rows": len(update),
        "n_calibration_score_rows": len(tables["calibration_scores"]),
        "n_initial_prediction_rows": len(baseline),
        "note": "Repeated rows are paired evaluations of fixed measurements, not independent experiments.",
    }
    write_json(ROOT / "verification/execution_checks.json", checks)
    files = {
        str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
        for p in sorted((ROOT / "results").glob("*.csv"))
    }
    write_json(ROOT / "verification/result_manifest.json", files)
    print("COMPLETE", json.dumps(checks), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
