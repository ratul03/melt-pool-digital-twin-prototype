"""Keyhole check for Section 7.1.3 (16 Nov 2026).
Usage: python code/keyhole_check.py <corrected study folder> <prepared depth.csv>
Reruns depth, primary cohort, budget 9 with the study code (common.py) and keeps one row per test record.
Check: equal-alloy interval scores must print 438.789 (ml_auto_jk), 451.603 (target_linPV_jk), 624.139 (target_split).
"""

import sys
import json
import numpy as np
import pandas as pd

ROOT = sys.argv[1]
sys.path.insert(0, ROOT + "/code")
from common import split_q, all_intervals

parts = [
    p
    for p in json.load(open(ROOT + "/protocol/partitions.json"))
    if p["task"] == "depth" and p["cohort"] == "primary"
]
pred = pd.read_csv(ROOT + "/data/ml_predictions.csv", float_precision="round_trip")
pred = pred[pred.task == "depth"]
src = pd.read_csv(ROOT + "/data/source_calibration_scores.csv", float_precision="round_trip")
reg = pd.read_csv(ROOT + "/data/model_registry.csv")
lab = pd.read_csv(sys.argv[2]).set_index("row_id")["meltpool shape"].fillna("unlabelled")
rows = []
for alloy in sorted({p["alloy"] for p in parts}):
    for seed in range(5):
        P = pred[(pred.alloy == alloy) & (pred.seed == seed)].set_index("row_id")
        q = split_q(
            src[(src.task == "depth") & (src.alloy == alloy) & (src.seed == seed)].score.to_numpy()
        )
        mu = float(
            reg[(reg.task == "depth") & (reg.alloy == alloy) & (reg.seed == seed)].train_mean.iloc[
                0
            ]
        )
        for part in [p for p in parts if p["alloy"] == alloy]:
            T = P.loc[part["test_row_ids"]]
            ti = dict(p=T.predicted.to_numpy(), X=T[["Power", "Velocity"]].to_numpy())
            y = T.actual.to_numpy()
            Cc = P.loc[part["calibration_order"][:9]]
            cal = dict(
                y=Cc.actual.to_numpy(),
                p=Cc.predicted.to_numpy(),
                X=Cc[["Power", "Velocity"]].to_numpy(),
            )
            res = all_intervals(cal, ti, q, mu)
            for m in ("ml_auto_jk", "target_split", "target_linPV_jk"):
                lo, hi, pt, ch = res[m]
                lo = np.maximum(lo, 0)
                hi = np.maximum(hi, 0)
                for k, rid in enumerate(part["test_row_ids"]):
                    rows.append(
                        (
                            alloy,
                            seed,
                            part["repeat"],
                            m,
                            rid,
                            y[k],
                            lo[k],
                            hi[k],
                            pt[k],
                            T.Power.iloc[k],
                            T.Velocity.iloc[k],
                        )
                    )
d = pd.DataFrame(rows, columns="alloy seed rep method row_id y lo hi pt P V".split())
d["label"] = d.row_id.map(lab)
d["up"] = 20 * np.clip(d.y - d.hi, 0, None)
d["low"] = 20 * np.clip(d.lo - d.y, 0, None)
d["IS"] = (d.hi - d.lo) + d.up + d.low
d["cov"] = (d.y >= d.lo) & (d.y <= d.hi)
d.to_csv("keyhole_check_records.csv", index=False)
g = (
    d.groupby(["method", "alloy", "seed", "rep"])
    .IS.mean()
    .groupby(["method", "alloy", "seed"])
    .mean()
    .groupby(["method", "alloy"])
    .mean()
    .groupby("method")
    .mean()
)
print(g)
# Summary by label for the proposed method
a = d[d.method == "ml_auto_jk"].copy()
a["miss_up"] = a.y > a.hi
for al, x in a.groupby("alloy"):
    lx = x[x.label != "unlabelled"]
    k = lx.label == "keyhole"
    if k.sum() == 0:
        print(al, "no keyhole-labelled test records")
        continue
    print(
        al,
        "labelled share",
        round(len(lx) / len(x), 3),
        "keyhole share of labelled",
        round(k.mean(), 3),
        "keyhole share of upper penalty (labelled)",
        round(lx[k].up.sum() / lx.up.sum(), 3),
        "above range: keyhole",
        round(lx[k].miss_up.mean(), 3),
        "other",
        round(lx[~k].miss_up.mean(), 3),
        "upper penalty from unlabelled",
        round(x[x.label == "unlabelled"].up.sum() / x.up.sum(), 3),
    )
