"""Corrected classification check: melting-temperature unit error (degC rows -> K).
Separate sensitivity analysis. Frozen results are not changed.
Same partitions, features, model settings and seeds as the frozen Weeks 1-2 / 20 Oct studies.
Step A: refit on ORIGINAL data and confirm exact agreement with the stored results (pipeline check).
Step B: refit on CORRECTED data and report the same metrics.
"""

import json
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

W12 = Path(sys.argv[1])
FINAL = Path(sys.argv[2])
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(W12 / "inputs/code"))
import reproduce as b

LABELS = b.LABELS
spec = json.loads((W12 / "protocol/frozen.json").read_text())
cfg = spec["tasks"]["classification"]
F = cfg["features"]
T = cfg["target"]
splits = json.loads((W12 / "protocol/splits.json").read_text())
future = [
    f
    for f in json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
    if f["task"] == "classification"
]
orig = pd.read_csv(W12 / "data/classification.csv", float_precision="round_trip").set_index(
    "row_id"
)

# Supplementary Table A.10, melting temperature in K (Akbari et al., 2022).
TABLE_A10_K = {
    "SS316L": 1688,
    "Ti-6Al-4V": 1922,
    "IN718": 1673,
    "AlSi10Mg": 1142,
    "Tungsten": 3695,
    "HCP Cu": 1631,
    "TiC/Inconel 718": 2015,
}
checks = {}
assert set(orig.alloy.unique()) <= set(TABLE_A10_K), set(orig.alloy.unique()) - set(TABLE_A10_K)
ref = orig.alloy.map(TABLE_A10_K)
is_c = np.isclose(orig["melting T"], ref - 273)
is_k = np.isclose(orig["melting T"], ref)
checks["every_row_is_K_or_K_minus_273"] = bool((is_c | is_k).all())
checks["rows_in_degC"] = int(is_c.sum())
cor = orig.copy()
cor.loc[is_c, "melting T"] = ref[is_c]
checks["corrected_one_value_per_alloy_equal_table"] = bool(
    (cor["melting T"] == cor.alloy.map(TABLE_A10_K)).all()
)
# Recompute absorptivity 1 with the frozen equation; first confirm it reproduces the stored values.
recalc_orig = b.abs1(orig, orig.eta_m_material)
checks["abs1_recompute_matches_stored_max_abs_diff"] = float(
    np.nanmax(np.abs(recalc_orig - orig.absorptivity_1))
)
cor["absorptivity_1"] = b.abs1(cor, cor.eta_m_material)
checks["abs1_changed_rows"] = int((~np.isclose(cor.absorptivity_1, orig.absorptivity_1)).sum())
checks["abs1_nan_after_correction"] = int(cor.absorptivity_1.isna().sum())
changed = orig[is_c].assign(
    melting_T_corrected=cor.loc[is_c, "melting T"],
    absorptivity_1_corrected=cor.loc[is_c, "absorptivity_1"],
)
changed[
    [
        "alloy",
        "source",
        "label",
        "melting T",
        "melting_T_corrected",
        "absorptivity_1",
        "absorptivity_1_corrected",
    ]
].to_csv(OUT / "corrected_rows.csv")


def fit_predict(d, s):
    tr, te = d.loc[s["train"]], d.loc[s["test"]]
    xs = StandardScaler().fit(tr[F].to_numpy())
    m = b.estimator("RFC", "classification", s["seed"], "corrected")
    m.fit(xs.transform(tr[F].to_numpy()), tr[T].to_numpy())

    def probs(rows):
        p = m.predict_proba(xs.transform(rows[F].to_numpy()))
        return np.column_stack(
            [
                p[:, list(m.classes_).index(l)] if l in m.classes_ else np.zeros(len(rows))
                for l in LABELS
            ]
        )

    return m, xs, probs


counts = pd.read_csv(W12 / "data/alloy_counts.csv")
eligible = set(counts.loc[(counts.task == "classification") & (counts.n >= 20), "alloy"])
transfer_rows, set_rows = [], []
max_prob_diff = 0.0
label_mismatch = 0
with threadpool_limits(limits=1):
    for i, s in enumerate(splits):
        if s["task"] != "classification":
            continue
        key = f"classification_{s['protocol']}_{i:03d}_s{s['seed']}"
        stored = pd.read_csv(W12 / f"results/runs/{key}_predictions.csv")
        for version, d in [("original", orig), ("corrected", cor)]:
            m, xs, probs = fit_predict(d, s)
            te = d.loc[s["test"]]
            pred = m.predict(xs.transform(te[F].to_numpy()))
            P = probs(te)
            if version == "original":
                label_mismatch += int((pred != stored.predicted.to_numpy()).sum())
                max_prob_diff = max(
                    max_prob_diff,
                    float(np.max(np.abs(P - stored[["prob_" + l for l in LABELS]].to_numpy()))),
                )
            frame = pd.DataFrame(
                {
                    "alloy": te.alloy.to_numpy(),
                    "actual": te[T].to_numpy(),
                    "ml": pred,
                    "simple": d.loc[s["train"], T].mode().iloc[0],
                }
            )
            for alloy, z in frame.groupby("alloy"):
                if alloy not in eligible:
                    continue
                for predictor in ["ml", "simple"]:
                    transfer_rows.append(
                        dict(
                            version=version,
                            protocol=s["protocol"],
                            model_seed=s["seed"],
                            alloy=alloy,
                            predictor=predictor,
                            n_test=len(z),
                            accuracy=float((z.actual == z[predictor]).mean()),
                            macro_f1=float(
                                f1_score(
                                    z.actual,
                                    z[predictor],
                                    labels=LABELS,
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                        )
                    )
            # Class sets (fixed-test calibration study design, 20 Oct): LOAO models for alloys with future splits.
            fs = [
                f
                for f in future
                if s["protocol"] == "leave_one_alloy_out" and f["alloy"] == s["alloy"]
            ]
            if not fs:
                continue
            cal = d.loc[s["calibration"]]
            Pcal = probs(cal)
            src_scores = 1 - Pcal[np.arange(len(cal)), [LABELS.index(v) for v in cal[T]]]
            for f in fs:
                test = d.loc[f["test_row_ids"]]
                Pt = probs(test)
                ptest = m.predict(xs.transform(test[F].to_numpy()))
                for budget in [0, 9, 20]:
                    ids = s["calibration"] if budget == 0 else f[f"calibration{budget}"]
                    if ids is None:
                        continue
                    if budget == 0:
                        sc = src_scores
                    else:
                        c = d.loc[ids]
                        Pc = probs(c)
                        sc = 1 - Pc[np.arange(len(c)), [LABELS.index(v) for v in c[T]]]
                    n = len(sc)
                    k = (9 * (n + 1) + 9) // 10
                    q = float(np.sort(sc)[k - 1]) if k <= n else math.inf
                    inc = (1 - Pt) <= q
                    sizes = inc.sum(1)
                    cov = np.mean([inc[j, LABELS.index(y)] for j, y in enumerate(test[T])])
                    set_rows.append(
                        dict(
                            version=version,
                            alloy=s["alloy"],
                            model_seed=s["seed"],
                            repeat=f["repeat"],
                            budget=budget,
                            cohort=(
                                "primary_paired"
                                if f["primary_budget_comparison"]
                                else "secondary_9_only"
                            ),
                            coverage=float(cov),
                            mean_set_size=float(sizes.mean()),
                            full_set_fraction=float(np.mean(sizes == 4)),
                            accuracy=float(np.mean(ptest == test[T].to_numpy())),
                        )
                    )
        print(key, "done", flush=True)

checks["original_refit_label_mismatches"] = label_mismatch
checks["original_refit_max_probability_difference"] = max_prob_diff
tr = pd.DataFrame(transfer_rows)
tr.to_csv(OUT / "transfer_per_alloy_seed.csv", index=False)
seed = (
    tr.groupby(["version", "protocol", "model_seed", "predictor"])[["accuracy", "macro_f1"]]
    .mean()
    .reset_index()
)
summ = (
    seed.groupby(["version", "protocol", "predictor"])[["accuracy", "macro_f1"]]
    .mean()
    .reset_index()
)
summ["n_alloys"] = tr.groupby(["version", "protocol", "predictor"]).alloy.nunique().to_numpy()
summ.to_csv(OUT / "transfer_summary.csv", index=False)
st = pd.DataFrame(set_rows)
st.to_csv(OUT / "class_sets_conditions.csv", index=False)
st = st[st.cohort == "primary_paired"]
a = (
    st.groupby(["version", "alloy", "model_seed", "budget"])[
        ["coverage", "mean_set_size", "full_set_fraction", "accuracy"]
    ]
    .mean()
    .reset_index()
)
a = (
    a.groupby(["version", "alloy", "budget"])[
        ["coverage", "mean_set_size", "full_set_fraction", "accuracy"]
    ]
    .mean()
    .reset_index()
)
a.to_csv(OUT / "class_sets_per_alloy.csv", index=False)
ss = (
    a.groupby(["version", "budget"])[["coverage", "mean_set_size", "full_set_fraction", "accuracy"]]
    .mean()
    .reset_index()
)
ss.to_csv(OUT / "class_sets_summary.csv", index=False)
# Compare the ORIGINAL refit against the stored 20 Oct and Weeks 1-2 summaries.
fin = pd.read_csv(FINAL / "results/summary.csv")
fin = fin[
    (fin.task == "classification") & (fin.cohort == "primary_paired") & (fin.predictor == "ml")
]
for _, r in fin.iterrows():
    o = ss[(ss.version == "original") & (ss.budget == r.budget)].iloc[0]
    for col in ["coverage", "mean_set_size", "full_set_fraction", "accuracy"]:
        checks[f"class_sets_original_matches_20oct_{col}_b{int(r.budget)}"] = bool(
            abs(o[col] - r[col]) < 1e-9
        )
ts = pd.read_csv(FINAL / "results/transfer_summary.csv")
ts = ts[ts.task == "classification"]
for _, r in ts.iterrows():
    o = summ[
        (summ.version == "original")
        & (summ.protocol == r.protocol)
        & (summ.predictor == r.predictor)
    ].iloc[0]
    checks[f"transfer_original_matches_20oct_{r.protocol}_{r.predictor}"] = bool(
        abs(o.accuracy - r.accuracy) < 1e-12
    )
(OUT / "checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
print(summ.round(4).to_string())
print(ss.round(4).to_string())
