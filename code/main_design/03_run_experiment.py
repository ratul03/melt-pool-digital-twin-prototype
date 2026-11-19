"""Run every method on every frozen partition. Usage: python 03_run_experiment.py <task> [alloy]

Writes one CSV per (task, alloy) under results/conditions/. Existing files are never overwritten.
"""
import json
import sys
import numpy as np
import pandas as pd
from common import ROOT, SRC, METHODS, sha, split_q, all_intervals, metrics

spec = json.loads((ROOT / "protocol/confirmatory_spec.json").read_text())
AMENDED = json.loads((ROOT / "protocol/correction_2026-11-02.json").read_text())["amended_files"]
for f, h in spec["frozen_files"].items():
    if f in AMENDED:
        assert sha(ROOT / f) == AMENDED[f]["new_sha256"], f"Amended file changed after correction: {f}"
        continue
    assert sha(ROOT / f) == h, f"Frozen file changed after freeze: {f}"
for f, h in spec["inherited_files"].items():
    assert sha(SRC / f) == h, f"Inherited file changed: {f}"

task = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
parts = [p for p in json.loads((ROOT / "protocol/partitions.json").read_text()) if p["task"] == task]
pred = pd.read_csv(ROOT / "data/ml_predictions.csv", float_precision="round_trip")
pred = pred[pred.task == task]
src = pd.read_csv(ROOT / "data/source_calibration_scores.csv", float_precision="round_trip")
reg = pd.read_csv(ROOT / "data/model_registry.csv")
outdir = ROOT / "results/conditions"
outdir.mkdir(parents=True, exist_ok=True)

for alloy in sorted({p["alloy"] for p in parts}):
    if only and alloy != only:
        continue
    dest = outdir / f"{task}__{alloy.replace('/', '_').replace(' ', '_')}.csv"
    if dest.exists():
        print("exists, skipped:", dest.name)
        continue
    rows = []
    for seed in range(5):
        P = pred[(pred.alloy == alloy) & (pred.seed == seed)].set_index("row_id")
        q_source = split_q(src[(src.task == task) & (src.alloy == alloy) & (src.seed == seed)].score.to_numpy())
        mu = float(reg[(reg.task == task) & (reg.alloy == alloy) & (reg.seed == seed)].train_mean.iloc[0])
        for part in [p for p in parts if p["alloy"] == alloy]:
            T = P.loc[part["test_row_ids"]]
            test_inputs = dict(p=T.predicted.to_numpy(), X=T[["Power", "Velocity"]].to_numpy())
            y_test = T.actual.to_numpy()          # used by metrics() only
            for b in part["budgets"]:
                C = P.loc[part["calibration_order"][:b]]
                cal = dict(y=C.actual.to_numpy(), p=C.predicted.to_numpy(), X=C[["Power", "Velocity"]].to_numpy())
                res = all_intervals(cal, test_inputs, q_source, mu)
                for name, (lo, hi, point, choice) in res.items():
                    for clip in (False, True):
                        # Correction 2026-11-02: project BOTH endpoints onto [0, inf).
                        # The frozen run clipped only the lower bound, which inverted intervals whose upper bound was negative.
                        lo_used = np.maximum(lo, 0.0) if clip else lo
                        hi_used = np.maximum(hi, 0.0) if clip else hi
                        m = metrics(y_test, lo_used, hi_used, point)
                        rows.append(dict(task=task, alloy=alloy, cohort=part["cohort"], seed=seed, repeat=part["repeat"],
                                         budget=b, method=name, family=METHODS[name][0], clipped=clip,
                                         choice="" if choice is None else choice, **m))
        print(task, alloy, "seed", seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print("wrote", dest.name, len(rows), flush=True)
