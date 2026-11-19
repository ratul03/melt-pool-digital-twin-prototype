from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
W12 = BASE / "Thesis_C_Weeks_1_2"
W34 = BASE / "Thesis_C_Weeks_3_4"
LABELS = ["LOF", "balling", "desirable", "keyhole"]
CONDITION = ["task", "cohort", "alloy", "model_seed", "repeat", "budget", "predictor"]
METRICS = [
    "coverage",
    "mean_width",
    "mean_interval_score",
    "MAE",
    "RMSE",
    "R2",
    "lower_miss_penalty",
    "upper_miss_penalty",
    "negative_point_fraction",
    "negative_lower_fraction",
    "accuracy",
    "macro_f1",
    "mean_set_size",
    "empty_set_fraction",
    "singleton_fraction",
    "full_set_fraction",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def read_csv(path, **kwargs):
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def quantile(scores):
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Invalid calibration scores")
    rank = (9 * (len(a) + 1) + 9) // 10
    return rank, float(np.sort(a)[rank - 1]) if rank <= len(a) else float("inf")


def geometry_metrics(g):
    y, p = g.actual.to_numpy(float), g.predicted.to_numpy(float)
    lo, hi = g.lower.to_numpy(float), g.upper.to_numpy(float)
    lower, upper = 20 * np.maximum(lo - y, 0), 20 * np.maximum(y - hi, 0)
    sst = np.sum((y - y.mean()) ** 2)
    # Preserve the evaluated inclusion decision through CSV round trips. The
    # independent verifier checks every decision, allowing only endpoint ulps.
    return dict(
        n_test=len(g),
        coverage=float(g.covered.mean()),
        mean_width=float(np.mean(hi - lo)),
        mean_interval_score=float(np.mean(hi - lo + lower + upper)),
        lower_miss_penalty=float(lower.mean()),
        upper_miss_penalty=float(upper.mean()),
        MAE=float(np.mean(abs(y - p))),
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
        R2=float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan,
        negative_point_fraction=float(np.mean(p < 0)),
        negative_lower_fraction=float(np.mean(lo < 0)),
    )


def classification_metrics(g):
    sets = [json.loads(s) for s in g.set_json]
    sizes = np.array([len(s) for s in sets])
    f1 = []
    for label in LABELS:
        tp = np.sum((g.actual == label) & (g.predicted == label))
        denom = np.sum(g.actual == label) + np.sum(g.predicted == label)
        f1.append(2 * tp / denom if denom else 0)
    return dict(
        n_test=len(g),
        coverage=float(np.mean([y in s for y, s in zip(g.actual, sets)])),
        accuracy=float(np.mean(g.actual == g.predicted)),
        macro_f1=float(np.mean(f1)),
        mean_set_size=float(sizes.mean()),
        empty_set_fraction=float(np.mean(sizes == 0)),
        singleton_fraction=float(np.mean(sizes == 1)),
        full_set_fraction=float(np.mean(sizes == 4)),
    )


def metric_table(predictions):
    rows = []
    for keys, g in predictions.groupby(CONDITION, sort=True):
        row = dict(zip(CONDITION, keys))
        row.update(
            classification_metrics(g) if row["task"] == "classification" else geometry_metrics(g)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(m, prefix=""):
    values = [c for c in METRICS if c in m]
    seedkeys = ["task", "cohort", "alloy", "model_seed", "budget", "predictor"]
    seed = m.groupby(seedkeys)[values].mean().reset_index()
    alloykeys = ["task", "cohort", "alloy", "budget", "predictor"]
    alloy = seed.groupby(alloykeys)[values].mean().reset_index()
    summarykeys = ["task", "cohort", "budget", "predictor"]
    summary = alloy.groupby(summarykeys)[values].mean().reset_index()
    summary["n_alloys"] = alloy.groupby(summarykeys).size().to_numpy()
    for name, table in [
        ("conditions", m),
        ("per_alloy_seed", seed),
        ("per_alloy", alloy),
        ("summary", summary),
    ]:
        table.to_csv(ROOT / "results" / f"{prefix}{name}.csv", index=False)
    return seed, alloy, summary
