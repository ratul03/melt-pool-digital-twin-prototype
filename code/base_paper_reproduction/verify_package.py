"""Independently recalculate scores from saved predictions and reload every model.
No fitting occurs. Nonphysical predictions are reported, never silently clipped.
Usage: python verify_package.py --reference-dir <earlier Thesis B package>
"""

from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, f1_score
from reproduce import ROOT, jobs, slug, load_data, CONSTANTS
from threadpoolctl import threadpool_limits


def main():
    checks = []
    registry = []
    comparison = []
    subgroups = []

    def check(name, passed, detail=""):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Earlier Thesis B package containing 06_RESULTS/csv.",
    )
    pkg = parser.parse_args().reference_dir
    ref = {
        m: pd.read_csv(pkg / "06_RESULTS/csv" / f)
        for m, f in [
            ("GB", "week6__gb_results.csv"),
            ("NN", "week6__nn_results.csv"),
            ("RF", "week6__rf_regression_results.csv"),
            ("RFC", "week6__rfc_classification_results.csv"),
        ]
    }
    for mode in ["historical", "corrected"]:
        r, c = load_data(mode)
        records = []
        for model, target, config in jobs():
            name = slug(model, target, config)
            out = ROOT / "results" / mode
            path = out / (name + ".json")
            check(mode + "/" + name + " exists", path.exists())
            if not path.exists():
                continue
            z = json.loads(path.read_text())
            records.append({k: v for k, v in z.items() if k not in ["features", "runs"]})
            ps = pd.read_csv(out / (name + "_predictions.csv"))
            splits = json.loads((out / (name + "_splits.json")).read_text())
            iscls = model == "RFC"
            metrics = ["accuracy", "auc", "macro_f1"] if iscls else ["R2", "MAE"]
            vals = {m: [] for m in metrics}
            data = (c if iscls else r).set_index("row_id")
            tc = (
                "label"
                if iscls
                else {
                    "depth": "depth of meltpool",
                    "width": "width of melt pool",
                    "length": "length of melt pool",
                }[target]
            )
            for seed, run in enumerate(z["runs"]):
                p = ps[ps.seed.eq(seed)]
                s = splits[seed]
                check(
                    mode + "/" + name + f" seed{seed} disjoint",
                    not set(s["train_row_ids"]) & set(s["test_row_ids"]),
                )
                check(
                    mode + "/" + name + f" seed{seed} row identities",
                    p.row_id.tolist() == s["test_row_ids"],
                )
                actual = data.loc[p.row_id, tc].to_numpy()
                check(
                    mode + "/" + name + f" seed{seed} targets",
                    (
                        np.array_equal(actual, p.actual.to_numpy())
                        if iscls
                        else np.allclose(
                            actual.astype(float), p.actual.to_numpy(), rtol=1e-12, atol=1e-12
                        )
                    ),
                )
                if iscls:
                    labels = ["LOF", "balling", "desirable", "keyhole"]
                    prob = p[["prob_" + cl for cl in labels]].to_numpy()
                    score = dict(
                        accuracy=accuracy_score(p.actual, p.predicted),
                        auc=roc_auc_score(
                            p.actual, prob, multi_class="ovr", average="macro", labels=labels
                        ),
                        macro_f1=f1_score(p.actual, p.predicted, average="macro"),
                    )
                    check(
                        mode + "/" + name + f" seed{seed} probabilities",
                        np.allclose(prob.sum(axis=1), 1),
                    )
                else:
                    score = dict(
                        R2=r2_score(p.actual, p.predicted),
                        MAE=mean_absolute_error(p.actual, p.predicted),
                    )
                for key, v in score.items():
                    vals[key].append(v)
                    check(mode + "/" + name + f" seed{seed} " + key, abs(v - run[key]) < 1e-8)
                if not iscls and config in ["baseline", "baseline+elemental", "baseline+hatch"]:
                    pp = p.copy()
                    pp["process"] = data.loc[p.row_id, "Process"].to_numpy()
                    for process, g in pp.groupby("process"):
                        subgroups.append(
                            dict(
                                mode=mode,
                                model=model,
                                target=target,
                                config=config,
                                seed=seed,
                                process=process,
                                n=len(g),
                                R2=r2_score(g.actual, g.predicted) if len(g) > 1 else None,
                                MAE=mean_absolute_error(g.actual, g.predicted),
                            )
                        )
            for key, v in vals.items():
                check(mode + "/" + name + " mean " + key, abs(np.mean(v) - z[key + "_mean"]) < 1e-8)
                check(
                    mode + "/" + name + " sample std " + key,
                    abs(np.std(v, ddof=1) - z[key + "_std"]) < 1e-8,
                )
            artifact_path = ROOT / "models" / mode / (name + ".joblib")
            a = joblib.load(artifact_path)
            check(mode + "/" + name + " feature order", a["features"] == z["features"])
            check(mode + "/" + name + " model seed", a["model"].random_state == 0)
            check(
                mode + "/" + name + " model family",
                a["model"].__class__.__name__
                == {
                    "GB": "GradientBoostingRegressor",
                    "RF": "RandomForestRegressor",
                    "RFC": "RandomForestClassifier",
                    "NN": "MLPRegressor",
                }[model],
            )
            if model == "NN":
                h, alpha = CONSTANTS["NN_PARAMS"][target]
                check(
                    mode + "/" + name + " neural hyperparameters",
                    a["model"].hidden_layer_sizes == tuple(h)
                    and a["model"].alpha == alpha
                    and a["model"].max_iter == (3000 if mode == "historical" else 20000),
                )
            else:
                check(
                    mode + "/" + name + " trees",
                    a["model"].n_estimators
                    == (
                        {"GB": CONSTANTS["GB_N_ESTIMATORS"].get(target), "RF": 500, "RFC": 160}[
                            model
                        ]
                    ),
                )
            X = data.loc[a["train_row_ids"], a["features"]].to_numpy(dtype=float)
            check(
                mode + "/" + name + " training-only input scaler",
                np.allclose(a["scaler"].mean_, X.mean(axis=0), rtol=1e-12, atol=1e-10),
            )
            if a["y_scaler"] is not None:
                check(
                    mode + "/" + name + " training-only target scaler",
                    np.allclose(
                        a["y_scaler"].mean_, data.loc[a["train_row_ids"], tc].astype(float).mean()
                    ),
                )
            p0 = ps[ps.seed.eq(0)]
            Xtest = data.loc[p0.row_id, a["features"]].to_numpy(dtype=float)
            pred = a["model"].predict(a["scaler"].transform(Xtest))
            if a["y_scaler"] is not None:
                pred = a["y_scaler"].inverse_transform(pred.reshape(-1, 1)).ravel()
            check(
                mode + "/" + name + " reloaded predictions",
                (
                    np.array_equal(pred, p0.predicted.to_numpy())
                    if iscls
                    else np.allclose(pred, p0.predicted, rtol=1e-10, atol=1e-8)
                ),
            )
            negatives = 0 if iscls else int((ps.predicted < 0).sum())
            warn = sum(len(run["warnings"]) for run in z["runs"])
            registry.append(
                dict(
                    mode=mode,
                    model=model,
                    target=target,
                    config=config,
                    n=z["n"],
                    path=str(artifact_path.relative_to(ROOT)),
                    usage=a["usage"],
                    negative_test_predictions_across_seeds=negatives,
                    training_warnings=warn,
                    largest_n_iter=max([run["n_iter"] or 0 for run in z["runs"]]),
                    verified_reload=True,
                    score_scope="five repeated held-out splits; seed-0 model saved",
                )
            )
            if mode == "historical":
                rr = ref[model]
                rr = rr[rr.config.eq(config)]
                if not iscls:
                    rr = rr[rr.target.eq(target)]
                for key in ["accuracy", "auc"] if iscls else ["R2", "MAE"]:
                    for stat in ["mean", "std"]:
                        metric = key + "_" + stat
                        expected = float(rr.iloc[0][metric])
                        observed = z[metric]
                        tol = 0.00501 if key == "MAE" else 0.000051
                        comparison.append(
                            dict(
                                model=model,
                                target=target,
                                config=config,
                                metric=metric,
                                reported=expected,
                                recomputed=observed,
                                delta=observed - expected,
                                tolerance=tol,
                                within_rounding=abs(observed - expected) <= tol,
                            )
                        )
        pd.DataFrame(records).to_csv(ROOT / "results" / mode / "summary.csv", index=False)
    pd.DataFrame(comparison).to_csv(
        ROOT / "audit/all_historical_results_comparison.csv", index=False
    )
    pd.DataFrame(registry).to_csv(ROOT / "models/MODEL_REGISTRY.csv", index=False)
    pd.DataFrame(subgroups).to_csv(ROOT / "audit/process_specific_scores.csv", index=False)
    result = dict(
        passed=sum(x["passed"] for x in checks),
        failed=sum(not x["passed"] for x in checks),
        model_count=len(registry),
        checks=checks,
    )
    (ROOT / "audit/package_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print({k: v for k, v in result.items() if k != "checks"})
    print(
        "Historical cells outside rounding:",
        sum(not x["within_rounding"] for x in comparison),
        "of",
        len(comparison),
    )
    if result["failed"]:
        print([x for x in checks if not x["passed"]])
        sys.exit(1)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
