from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "protocol/frozen.json").read_text())
splits = json.loads((ROOT / "protocol/splits.json").read_text())
labels = ["LOF", "balling", "desirable", "keyhole"]
checks = []
rows = []
predictions = []
registry = []


def check(name, condition):
    checks.append(dict(check=name, passed=bool(condition)))
    if not condition:
        print("FAIL", name)


def metrics(y, p, cls):
    if cls:
        return dict(
            accuracy=accuracy_score(y, p),
            macro_f1=f1_score(y, p, labels=labels, average="macro", zero_division=0),
            balanced_accuracy=recall_score(
                y, p, labels=sorted(set(y)), average="macro", zero_division=0
            ),
        )
    return dict(
        R2=r2_score(y, p) if len(y) > 1 and np.var(y) > 0 else None,
        MAE=mean_absolute_error(y, p),
        RMSE=np.sqrt(mean_squared_error(y, p)),
        negative_predictions=int(np.sum(p < 0)),
    )


for rel, h in spec["input_hashes"].items():
    check("Frozen input " + rel, hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h)
data = {t: pd.read_csv(ROOT / "data" / f"{t}.csv").set_index("row_id") for t in spec["tasks"]}
raw = {
    t: pd.read_csv(ROOT / "inputs/data/raw" / ("meltpoolnet_" + s["file"] + ".csv"))
    for t, s in spec["tasks"].items()
}
for t, d in data.items():
    cfg = spec["tasks"][t]
    check(t + " SLM only", d["Sub-process"].eq("SLM").all())
    check(t + " unique raw identities", d.index.is_unique)
    check(
        t + " pre-build features",
        not set(cfg["features"])
        & {
            "absorptivity_2",
            "width of melt pool",
            "depth of meltpool",
            "length of melt pool",
            "label",
            "meltpool shape",
        },
    )
    if t != "classification":
        check(
            t + " raw targets unchanged",
            np.allclose(pd.to_numeric(raw[t].loc[d.index, cfg["target"]]), d[cfg["target"]]),
        )
        check(
            t + " suspect rows excluded",
            not set(d.index) & set(list(range(580, 583)) + list(range(837, 845))),
        )
    else:
        check(
            "classification raw labels unchanged",
            np.array_equal(raw[t].loc[d.index, "meltpool shape"].str.strip(), d.label),
        )
        exponent = (
            0.6
            * d.eta_m_material
            * d.Power
            / (
                (d["melting T"] - 298)
                * np.pi
                * d.density
                * d.Cp
                * (d.Velocity * 0.001)
                * (d["beam D"] * 0.0000005) ** 2
            )
        )
        expected = 0.7 * (1 - np.exp(-exponent))
        check(
            "classification equation2 independent calculation",
            np.allclose(d.absorptivity_1, expected, rtol=1e-10, atol=1e-12),
        )
        check("classification physical feature range", d.absorptivity_1.between(0, 0.7).all())

for k, s in enumerate(splits):
    key = f"{s['task']}_{s['protocol']}_{k:03d}_s{s['seed']}"
    path = ROOT / "results/runs" / f"{key}.json"
    check(key + " completed", path.exists())
    if not path.exists():
        continue
    r = json.loads(path.read_text())
    p = pd.read_csv(path.with_name(key + "_predictions.csv"))
    t = s["task"]
    c = t == "classification"
    d = data[t]
    cfg = spec["tasks"][t]
    tr, ca, te = [set(s[x]) for x in ["train", "calibration", "test"]]
    check(key + " partitions disjoint", not (tr & ca or tr & te or ca & te))
    check(key + " partitions exhaustive", tr | ca | te == set(d.index))
    check(key + " prediction row order", p.row_id.tolist() == s["test"])
    check(
        key + " prediction labels",
        (
            np.array_equal(p.actual, d.loc[s["test"], cfg["target"]])
            if c
            else np.allclose(p.actual, d.loc[s["test"], cfg["target"]])
        ),
    )
    check(
        key + " train-only feature scaling",
        np.allclose(r["scaler_mean"], d.loc[s["train"], cfg["features"]].mean().to_numpy()),
    )
    if cfg["target_scaling"]:
        check(
            key + " train-only target scaling",
            np.isclose(r["target_scaler_mean"], d.loc[s["train"], cfg["target"]].mean()),
        )
    if s["protocol"] == "leave_one_alloy_out":
        check(
            key + " full alloy exclusion",
            set(d.loc[list(te), "alloy"]).isdisjoint(set(d.loc[list(tr | ca), "alloy"])),
        )
        check(key + " all test alloy rows", te == set(d.index[d.alloy.eq(s["alloy"])]))
    for metric, value in metrics(p.actual.to_numpy(), p.predicted.to_numpy(), c).items():
        check(
            key + " metric " + metric,
            (
                r["metrics"][metric] is None
                if value is None
                else np.isclose(value, r["metrics"][metric], rtol=1e-9, atol=1e-9)
            ),
        )
    for metric, value in metrics(p.actual.to_numpy(), p.dummy.to_numpy(), c).items():
        check(
            key + " dummy " + metric,
            (
                r["dummy_metrics"][metric] is None
                if value is None
                else np.isclose(value, r["dummy_metrics"][metric], rtol=1e-9, atol=1e-9)
            ),
        )
    if c:
        check(
            key + " probability sums", np.allclose(p[["prob_" + x for x in labels]].sum(axis=1), 1)
        )
        check(
            key + " confusion matrix",
            np.array_equal(
                confusion_matrix(p.actual, p.predicted, labels=labels), r["confusion_matrix"]
            ),
        )
    if "saved_model" in r:
        art = joblib.load(ROOT / r["saved_model"])
        q = art["model"].predict(
            art["scaler"].transform(d.loc[s["test"], cfg["features"]].to_numpy())
        )
        if art["target_scaler"] is not None:
            q = art["target_scaler"].inverse_transform(q.reshape(-1, 1)).ravel()
        check(
            key + " independent artifact reload",
            (
                np.array_equal(q, p.predicted)
                if c
                else np.allclose(q, p.predicted, rtol=1e-12, atol=1e-9)
            ),
        )
        check(key + " artifact feature order", art["features"] == cfg["features"])
        check(
            key + " protocol provenance",
            art["protocol_sha256"]
            == hashlib.sha256((ROOT / "protocol/frozen.json").read_bytes()).hexdigest(),
        )
        actualparams = art["model"].get_params()
        check(
            key + " frozen parameters",
            all(
                actualparams[x] == (tuple(v) if isinstance(actualparams[x], tuple) else v)
                for x, v in cfg["parameters"].items()
            ),
        )
        registry.append(
            dict(
                task=t,
                protocol=s["protocol"],
                alloy=s["alloy"],
                path=r["saved_model"],
                n_train=r["n_train"],
                n_reserved_cal=r["n_calibration"],
                n_test=r["n_test"],
                reload_verified=True,
            )
        )
    rows.append(
        {
            x: r[x]
            for x in [
                "key",
                "task",
                "protocol",
                "seed",
                "alloy",
                "primary",
                "n_train",
                "n_calibration",
                "n_test",
                "iterations",
            ]
        }
        | r["metrics"]
        | {"dummy_" + x: v for x, v in r["dummy_metrics"].items()}
        | dict(
            warning_count=len(r["warnings"]),
            warnings=" | ".join(r["warnings"]),
            missing_train_classes="|".join(r["missing_train_classes"]),
        )
    )
    p["task"] = t
    p["protocol"] = s["protocol"]
    p["seed"] = s["seed"]
    predictions.append(p)

future = json.loads((ROOT / "protocol/future_calibration_splits.json").read_text())
for i, s in enumerate(future):
    d = data[s["task"]]
    test = set(s["test_row_ids"])
    pool = set(s["pool_row_ids"])
    check(
        f"future {i} complete disjoint",
        not test & pool and test | pool == set(d.index[d.alloy.eq(s["alloy"])]),
    )
    check(
        f"future {i} 9 labels separate",
        len(set(s["calibration9"])) == 9 and set(s["calibration9"]) <= pool,
    )
    if s["calibration20"] is not None:
        check(
            f"future {i} nested 20 labels separate",
            len(set(s["calibration20"])) == 20
            and set(s["calibration9"]) <= set(s["calibration20"]) <= pool,
        )

result = dict(
    passed=sum(x["passed"] for x in checks),
    failed=sum(not x["passed"] for x in checks),
    expected_runs=len(splits),
    completed_runs=len(rows),
    saved_models=len(registry),
    checks=checks,
)
(ROOT / "verification/checks.json").write_text(json.dumps(result, indent=2))
assert result["failed"] == 0, result
pd.DataFrame(rows).to_csv(ROOT / "results/per_run_metrics.csv", index=False)
pd.DataFrame(registry).to_csv(ROOT / "models/REGISTRY.csv", index=False)
allp = pd.concat(predictions, ignore_index=True)
allp.to_csv(ROOT / "results/all_predictions.csv", index=False)
counts = pd.read_csv(ROOT / "data/alloy_counts.csv")
seedsummary = []
groupmetrics = []
for (t, protocol, seed), g in allp.groupby(["task", "protocol", "seed"]):
    primary = counts.loc[counts.task.eq(t) & counts.primary, "alloy"].tolist()
    g = g[g.alloy.isin(primary)]
    vals = metrics(g.actual.to_numpy(), g.predicted.to_numpy(), t == "classification")
    dummy = metrics(g.actual.to_numpy(), g.dummy.to_numpy(), t == "classification")
    seedsummary.append(
        dict(
            task=t,
            protocol=protocol,
            seed=seed,
            n_test=len(g),
            alloys=g.alloy.nunique(),
            **vals,
            **{"dummy_" + k: v for k, v in dummy.items()},
        )
    )
    for alloy, h in g.groupby("alloy"):
        groupmetrics.append(
            dict(
                task=t,
                protocol=protocol,
                seed=seed,
                alloy=alloy,
                n_test=len(h),
                **metrics(h.actual.to_numpy(), h.predicted.to_numpy(), t == "classification"),
            )
        )
ss = pd.DataFrame(seedsummary)
ss.to_csv(ROOT / "results/primary_cohort_per_seed.csv", index=False)
gg = pd.DataFrame(groupmetrics)
gg.to_csv(ROOT / "results/primary_per_alloy_per_seed.csv", index=False)
numeric = [x for x in ss.select_dtypes(include="number").columns if x != "seed"]
summary = ss.groupby(["task", "protocol"])[numeric].agg(["mean", "std"])
summary.columns = ["_".join(x) for x in summary.columns]
summary.reset_index().to_csv(ROOT / "results/summary.csv", index=False)
gsummary = gg.groupby(["task", "protocol", "alloy"])[
    [x for x in gg.select_dtypes(include="number").columns if x != "seed"]
].agg(["mean", "std"])
gsummary.columns = ["_".join(x) for x in gsummary.columns]
gsummary.reset_index().to_csv(ROOT / "results/per_alloy_summary.csv", index=False)
# Label row/seed repetition explicitly; matrix is 5 repeats, not independent new observations.
cs = allp[
    allp.task.eq("classification")
    & allp.protocol.eq("leave_one_alloy_out")
    & allp.alloy.isin(counts.loc[counts.task.eq("classification") & counts.primary, "alloy"])
]
pd.DataFrame(
    confusion_matrix(cs.actual, cs.predicted, labels=labels), index=labels, columns=labels
).to_csv(ROOT / "results/classification_confusion_5_repeats.csv")
print({k: v for k, v in result.items() if k != "checks"})
print(summary.to_string())
