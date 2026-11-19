"""Independent verification. Uses its own brute-force implementation; imports only paths and the frozen method code
for the label-shuffle test. Any failure exits non-zero.
"""

import itertools
import json
import math
import sys
import numpy as np
import pandas as pd
from common import ROOT, SRC, W12, LEAKY_COLUMNS, sha, write_json, all_intervals, split_q

rng = np.random.default_rng(99)
checks = []


def check(name, ok, ctx=""):
    checks.append(dict(check=name, passed=bool(ok), context=str(ctx)[:300]))


spec = json.loads((ROOT / "protocol/confirmatory_spec.json").read_text())
AMENDED = json.loads((ROOT / "protocol/correction_2026-11-02.json").read_text())["amended_files"]
for f, h in spec["frozen_files"].items():
    if f in AMENDED:
        check(
            "amended_file_matches_correction_record",
            sha(ROOT / f) == AMENDED[f]["new_sha256"] and AMENDED[f]["frozen_sha256"] == h,
            f,
        )
    else:
        check("frozen_file_unchanged", sha(ROOT / f) == h, f)
for f, h in spec["inherited_files"].items():
    check("inherited_file_unchanged", sha(SRC / f) == h, f)
frozen = json.loads((W12 / "protocol/frozen.json").read_text())
splits = json.loads((W12 / "protocol/splits.json").read_text())
old_future = json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
parts = json.loads((ROOT / "protocol/partitions.json").read_text())
pred = pd.read_csv(ROOT / "data/ml_predictions.csv", float_precision="round_trip")
src = pd.read_csv(ROOT / "data/source_calibration_scores.csv", float_precision="round_trip")
reg = pd.read_csv(ROOT / "data/model_registry.csv")
data = {
    t: pd.read_csv(W12 / f"data/{t}.csv", low_memory=False, float_precision="round_trip").set_index(
        "row_id"
    )
    for t in ["depth", "width", "length"]
}

# 1. Features and data
for t, d in data.items():
    f = frozen["tasks"][t]["features"]
    check("no_leaky_feature", not set(f) & LEAKY_COLUMNS, t)
    check("targets_nonnegative", (d[frozen["tasks"][t]["target"]] >= 0).all(), t)

# 2. Partitions: separation, duplicates (exact and tolerance), whole-alloy exclusion from models
overlap_old = []
for p in parts:
    t, a = p["task"], p["alloy"]
    d = data[t]
    f = frozen["tasks"][t]["features"]
    te, po = set(p["test_row_ids"]), set(p["pool_row_ids"])
    ctx = (t, a, p["repeat"])
    check("test_pool_disjoint", not te & po, ctx)
    check("test_pool_cover_alloy", te | po == set(d.index[d.alloy == a]), ctx)
    check(
        "calibration_order_is_pool_permutation", sorted(p["calibration_order"]) == sorted(po), ctx
    )
    check("test_size", 15 <= len(te) <= 20, ctx)
    check("budgets_fit_pool", all(b <= len(po) for b in p["budgets"]), ctx)
    T, P = d.loc[sorted(te), f].to_numpy(float), d.loc[sorted(po), f].to_numpy(float)
    close = np.isclose(T[:, None, :], P[None, :, :], rtol=1e-6, atol=1e-9).all(axis=2)
    check("no_test_condition_in_pool_tolerance_1e-6", not close.any(), ctx)
    old = [o for o in old_future if o["task"] == t and o["alloy"] == a]
    if old:
        overlap_old.append(len(te & set(old[0]["test_row_ids"])) / len(te))
for t, a in {(p["task"], p["alloy"]) for p in parts}:
    for s in splits:
        if s["task"] == t and s["alloy"] == a and s["protocol"] == "leave_one_alloy_out":
            d = data[t]
            check(
                "alloy_absent_from_model_training_and_source_cal",
                a not in set(d.loc[s["train"] + s["calibration"], "alloy"]),
                (t, a, s["seed"]),
            )

# 3. Cache equals frozen data and stored Weeks 1-2 predictions (independent read)
for (t, a, seed), g in pred.groupby(["task", "alloy", "seed"]):
    d = data[t]
    tc = frozen["tasks"][t]["target"]
    check("cache_actual_equals_data", np.allclose(g.actual, d.loc[g.row_id, tc]), (t, a, seed))
    key = reg[(reg.task == t) & (reg.alloy == a) & (reg.seed == seed)].key.iloc[0]
    old = pd.read_csv(
        W12 / f"results/runs/{key}_predictions.csv", float_precision="round_trip"
    ).set_index("row_id")
    check(
        "cache_pred_equals_weeks12",
        np.allclose(g.predicted.to_numpy(), old.loc[g.row_id, "predicted"], rtol=1e-9, atol=1e-7),
        key,
    )
    s = [
        x
        for x in splits
        if x["task"] == t
        and x["alloy"] == a
        and x["seed"] == seed
        and x["protocol"] == "leave_one_alloy_out"
    ][0]
    sc = src[(src.task == t) & (src.alloy == a) & (src.seed == seed)]
    check(
        "source_scores_cover_source_calibration_rows",
        sorted(sc.row_id) == sorted(s["calibration"]),
        key,
    )
    check(
        "train_mean_from_training_rows",
        np.isclose(reg[reg.key == key].train_mean.iloc[0], d.loc[s["train"], tc].mean()),
        key,
    )


# 4. Brute-force reimplementation (no hat matrix, explicit refits, explicit ranks)
def X_of(name, p, PV):
    one = np.ones((len(p), 1))
    return {
        "shift": one,
        "mean": one,
        "linear": np.column_stack([one, p]),
        "linE": np.column_stack([one, PV[:, 0] / PV[:, 1]]),
        "linPV": np.column_stack([one, PV]),
    }[name]


def fit(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


def loo_mse_brute(name, p, PV, t):
    n = len(t)
    errs = []
    for j in range(n):
        m = np.arange(n) != j
        A = X_of(name, p[m], PV[m])
        if np.linalg.matrix_rank(X_of(name, p, PV)) > np.linalg.matrix_rank(A):
            return np.inf  # removing j loses a direction = leverage 1
        errs.append((t[j] - X_of(name, p[j : j + 1], PV[j : j + 1]) @ fit(A, t[m]))[0] ** 2)
    return float(np.mean(errs))


def jk_brute(y, p, PV, pt, PVt, cands, residual):
    n = len(y)
    t = y - p if residual else y
    lows, highs = [], []
    for i in range(n):
        m = np.arange(n) != i
        c = (
            cands[int(np.argmin([loo_mse_brute(c, p[m], PV[m], t[m]) for c in cands]))]
            if len(cands) > 1
            else cands[0]
        )
        b = fit(X_of(c, p[m], PV[m]), t[m])
        fi = (X_of(c, p[i : i + 1], PV[i : i + 1]) @ b)[0] + (p[i] if residual else 0)
        R = abs(y[i] - fi)
        mu = X_of(c, pt, PVt) @ b + (pt if residual else 0)
        lows.append(mu - R)
        highs.append(mu + R)
    L, H = np.array(lows).T, np.array(highs).T
    lo_idx = math.floor(0.1 * (n + 1) + 1e-12) - 1
    hi_idx = math.ceil(0.9 * (n + 1) - 1e-12) - 1
    c = (
        cands[int(np.argmin([loo_mse_brute(c, p, PV, t) for c in cands]))]
        if len(cands) > 1
        else cands[0]
    )
    point = X_of(c, pt, PVt) @ fit(X_of(c, p, PV), t) + (pt if residual else 0)
    return np.sort(L, 1)[:, lo_idx], np.sort(H, 1)[:, hi_idx], point, c


def q_brute(s):
    s = sorted(s)
    k = math.ceil(0.9 * (len(s) + 1) - 1e-12)
    return s[k - 1] if k <= len(s) else math.inf


def metrics_brute(y, lo, hi, point):
    cov, isc, wid, mae = [], [], [], []
    for yy, l, h, pp in zip(y, lo, hi, point):
        cov.append(l - 1e-9 * max(1, abs(l)) <= yy <= h + 1e-9 * max(1, abs(h)))
        isc.append((h - l) + 20 * max(l - yy, 0) + 20 * max(yy - h, 0))
        wid.append(h - l)
        mae.append(abs(yy - pp))
    return dict(
        coverage=np.mean(cov),
        interval_score=np.mean(isc),
        mean_width=np.mean(wid),
        MAE=np.mean(mae),
    )


cond = pd.concat(
    [pd.read_csv(f) for f in sorted((ROOT / "results/conditions").glob("*.csv"))], ignore_index=True
)
cond["choice"] = cond.choice.fillna("")
CI = cond.set_index(["task", "alloy", "seed", "repeat", "budget", "method", "clipped"])
pidx = {(p["task"], p["alloy"], p["repeat"]): p for p in parts}
sample_keys = (
    cond[["task", "alloy", "seed", "repeat", "budget"]]
    .drop_duplicates()
    .sample(200, random_state=7)
)
spec_methods = {
    "target_mean_jk": (["mean"], False),
    "target_linE_jk": (["linE"], False),
    "target_linPV_jk": (["linPV"], False),
    "target_auto_jk": (["mean", "linE", "linPV"], False),
    "ml_shift_jk": (["shift"], True),
    "ml_linear_jk": (["linear"], True),
    "ml_auto_jk": (["shift", "linear"], True),
}
maxdiff = 0.0
for _, k in sample_keys.iterrows():
    p = pidx[(k.task, k.alloy, k["repeat"])]
    P = pred[(pred.task == k.task) & (pred.alloy == k.alloy) & (pred.seed == k.seed)].set_index(
        "row_id"
    )
    T, C = P.loc[p["test_row_ids"]], P.loc[p["calibration_order"][: k.budget]]
    y, pp, PV = C.actual.to_numpy(), C.predicted.to_numpy(), C[["Power", "Velocity"]].to_numpy()
    yt, pt, PVt = T.actual.to_numpy(), T.predicted.to_numpy(), T[["Power", "Velocity"]].to_numpy()
    qs = q_brute(
        src[(src.task == k.task) & (src.alloy == k.alloy) & (src.seed == k.seed)].score.tolist()
    )
    mu = reg[(reg.task == k.task) & (reg.alloy == k.alloy) & (reg.seed == k.seed)].train_mean.iloc[
        0
    ]
    res = {
        "source_split": (pt - qs, pt + qs, pt, ""),
        "target_split": (pt - q_brute(abs(y - pp)), pt + q_brute(abs(y - pp)), pt, ""),
        "mean_target_split": (
            np.full(len(pt), mu) - q_brute(abs(y - mu)),
            np.full(len(pt), mu) + q_brute(abs(y - mu)),
            np.full(len(pt), mu),
            "",
        ),
    }
    for name, (cands, resid) in spec_methods.items():
        lo, hi, point, c = jk_brute(y, pp, PV, pt, PVt, cands, resid)
        res[name] = (lo, hi, point, c if len(cands) > 1 else cands[0])
    for name, (lo, hi, point, c) in res.items():
        for clip in (False, True):
            lo_u, hi_u = (np.maximum(lo, 0), np.maximum(hi, 0)) if clip else (lo, hi)
            check(
                "interval_ordered_every_record",
                bool(np.all(lo_u <= hi_u + 1e-9 * np.maximum(1, np.abs(hi_u)))),
                (k.task, k.alloy, name, clip),
            )
            if clip:
                check(
                    "clipped_bounds_nonnegative",
                    bool(np.all(lo_u >= 0) and np.all(hi_u >= 0)),
                    (k.task, k.alloy, name),
                )
            mb = metrics_brute(yt, lo_u, hi_u, point)
            row = CI.loc[(k.task, k.alloy, k.seed, k["repeat"], k.budget, name, clip)]
            for kk, v in mb.items():
                diff = abs(row[kk] - v) / max(1.0, abs(v))
                maxdiff = max(maxdiff, diff)
                check(
                    "brute_force_metric_match",
                    diff < 1e-8,
                    (k.task, k.alloy, k.seed, k["repeat"], k.budget, name, clip, kk, row[kk], v),
                )
            if name in ("ml_auto_jk", "target_auto_jk"):
                check("brute_force_auto_choice_match", row["choice"] == c, (name, row["choice"], c))

# 5. Label-shuffle leakage test: change test labels -> identical bounds; change calibration labels -> bounds move
moved = 0
for _, k in sample_keys.head(60).iterrows():
    p = pidx[(k.task, k.alloy, k["repeat"])]
    P = (
        pred[(pred.task == k.task) & (pred.alloy == k.alloy) & (pred.seed == k.seed)]
        .set_index("row_id")
        .copy()
    )
    qs = split_q(
        src[(src.task == k.task) & (src.alloy == k.alloy) & (src.seed == k.seed)].score.to_numpy()
    )
    mu = reg[(reg.task == k.task) & (reg.alloy == k.alloy) & (reg.seed == k.seed)].train_mean.iloc[
        0
    ]

    def run(frame):
        T, C = frame.loc[p["test_row_ids"]], frame.loc[p["calibration_order"][: k.budget]]
        return all_intervals(
            dict(
                y=C.actual.to_numpy(),
                p=C.predicted.to_numpy(),
                X=C[["Power", "Velocity"]].to_numpy(),
            ),
            dict(p=T.predicted.to_numpy(), X=T[["Power", "Velocity"]].to_numpy()),
            qs,
            mu,
        )

    base = run(P)
    Q = P.copy()
    Q.loc[p["test_row_ids"], "actual"] = rng.uniform(0, 5000, len(p["test_row_ids"]))
    Q.loc[
        [
            r
            for r in P.index
            if r not in set(p["test_row_ids"]) | set(p["calibration_order"][: k.budget])
        ],
        "actual",
    ] = -1.0
    shuffled = run(Q)
    same = all(
        np.array_equal(base[m][0], shuffled[m][0]) and np.array_equal(base[m][1], shuffled[m][1])
        for m in base
    )
    check("test_and_unused_labels_do_not_change_intervals", same, tuple(k))
    R = P.copy()
    cal = p["calibration_order"][: k.budget]
    R.loc[cal, "actual"] = R.loc[cal, "actual"].to_numpy() * 1.5 + 10
    moved += int(not np.array_equal(run(R)["ml_auto_jk"][1], base["ml_auto_jk"][1]))
check("calibration_labels_do_change_intervals (sanity)", moved == 60, moved)

# 6a. Correction 2026-11-02: no condition may have a negative mean width (the runner also refuses any inverted record).
check(
    "no_negative_mean_width_any_condition",
    bool((cond.mean_width >= -1e-9).all()),
    int((cond.mean_width < -1e-9).sum()),
)
# 6. Clip validity: clipping never lowers coverage and never raises the interval score
pair = cond.pivot_table(
    index=["task", "alloy", "seed", "repeat", "budget", "method"],
    columns="clipped",
    values=["coverage", "interval_score"],
)
check("clip_keeps_coverage", np.allclose(pair["coverage"][True], pair["coverage"][False]))
check(
    "clip_never_raises_score",
    (pair["interval_score"][True] <= pair["interval_score"][False] + 1e-9).all(),
)

# 7. Aggregation recomputed with plain loops
agg = {}
for r in cond.itertuples():
    agg.setdefault((r.task, r.cohort, r.budget, r.method, r.clipped, r.alloy, r.seed), []).append(
        r.interval_score
    )
lvl2 = {}
for (t, c, b, m, cl, a, s), v in agg.items():
    lvl2.setdefault((t, c, b, m, cl, a), []).append(np.mean(v))
lvl3 = {}
for (t, c, b, m, cl, a), v in lvl2.items():
    lvl3.setdefault((t, c, b, m, cl), []).append(np.mean(v))
S = pd.read_csv(ROOT / "results/summary.csv").set_index(
    ["task", "cohort", "budget", "method", "clipped"]
)
for kk, v in lvl3.items():
    check(
        "summary_interval_score_recomputed",
        np.isclose(S.loc[kk, "interval_score"], np.mean(v), rtol=1e-10),
        kk,
    )

# 8. Exact Wilcoxon recomputed by enumerating all sign flips (primary hypothesis)
A = pd.read_csv(ROOT / "results/per_alloy.csv")
A = A[(A.task == "depth") & (A.cohort == "primary") & (A.budget == 9) & A.clipped].pivot(
    index="alloy", columns="method", values="interval_score"
)
diff = (A["ml_auto_jk"] - A["target_split"]).to_numpy()
ranks = pd.Series(np.abs(diff)).rank().to_numpy()
W = ranks[diff > 0].sum()
null = [
    sum(r for r, sgn in zip(ranks, s) if sgn) for s in itertools.product([0, 1], repeat=len(ranks))
]
mean_null = ranks.sum() / 2
p_exact = np.mean([abs(x - mean_null) >= abs(W - mean_null) - 1e-12 for x in null])
H = json.loads((ROOT / "results/primary_hypothesis.json").read_text())
check(
    "primary_wilcoxon_recomputed",
    np.isclose(p_exact, H["wilcoxon_exact_p"]),
    (p_exact, H["wilcoxon_exact_p"]),
)
check(
    "primary_mean_IS_recomputed",
    np.isclose(A["ml_auto_jk"].mean(), H["proposed_IS"])
    and np.isclose(A["target_split"].mean(), H["comparator_IS"]),
)

fails = [c for c in checks if not c["passed"]]
out = dict(
    passed=len(checks) - len(fails),
    failed=len(fails),
    failures=fails[:50],
    counts=pd.Series([c["check"] for c in checks]).value_counts().to_dict(),
    brute_force_max_relative_difference=maxdiff,
    new_test_rows_shared_with_20oct_fixed_test_mean_fraction=(
        float(np.mean(overlap_old)) if overlap_old else None
    ),
)
write_json(ROOT / "verification/verification.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=1))
print("FAILURES:", fails[:10])
sys.exit(1 if fails else 0)
