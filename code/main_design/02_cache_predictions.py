"""Reload the frozen excluded-alloy models (5 seeds), check them, cache predictions.

Checks per model (any failure stops the run):
- feature list equals the frozen spec and contains no post-build or target column;
- no target-alloy row in training or source calibration;
- feature scaler (and depth target scaler) equal training-row statistics only;
- reproduced predictions equal the stored Weeks 1-2 predictions.
"""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import ROOT, W12, W34, TASKS, LEAKY_COLUMNS, SUSPECT_IDS, write_json

spec = json.loads((W12 / "protocol/frozen.json").read_text())
splits = json.loads((W12 / "protocol/splits.json").read_text())
parts = json.loads((ROOT / "protocol/partitions.json").read_text())
need = sorted({(p["task"], p["alloy"]) for p in parts})
pred_rows, src_rows, meta, checks = [], [], [], []


def check(name, ok, ctx):
    checks.append(dict(check=name, passed=bool(ok), context=str(ctx)))
    if not ok:
        raise AssertionError(f"{name} failed: {ctx}")


with threadpool_limits(1):
    for task in TASKS:
        d = pd.read_csv(W12 / f"data/{task}.csv", low_memory=False, float_precision="round_trip").set_index("row_id")
        cfg = spec["tasks"][task]
        f, tc = cfg["features"], cfg["target"]
        check("no_suspect_unit_rows", not (set(d.index) & SUSPECT_IDS), task)
        check("slm_only", d["Sub-process"].eq("SLM").all(), task)
        check("targets_nonnegative", (d[tc] >= 0).all(), task)
        check("no_leaky_features", not (set(f) & LEAKY_COLUMNS), (task, f))
        for t2, alloy in [x for x in need if x[0] == task]:
            for i, s in enumerate(splits):
                if s["task"] != task or s["protocol"] != "leave_one_alloy_out" or s["alloy"] != alloy:
                    continue
                seed = s["seed"]
                key = f"{task}_leave_one_alloy_out_{i:03d}_s{seed}"
                path = (W12 if seed == 0 else W34) / "models" / f"{key}.joblib"
                art = joblib.load(path)
                ctx = (task, alloy, seed)
                check("feature_order", list(art["features"]) == list(f), ctx)
                tr, cal = d.loc[s["train"]], d.loc[s["calibration"]]
                check("target_alloy_not_in_training", alloy not in set(tr.alloy), ctx)
                check("target_alloy_not_in_source_calibration", alloy not in set(cal.alloy), ctx)
                check("test_is_whole_alloy", set(s["test"]) == set(d.index[d.alloy == alloy]), ctx)
                sc = art["scaler"]
                std = tr[f].std(ddof=0).to_numpy()
                check("scaler_mean_training_only", np.allclose(sc.mean_, tr[f].mean().to_numpy(), rtol=1e-10, atol=1e-12), ctx)
                check("scaler_scale_training_only", np.allclose(sc.scale_, np.where(std == 0, 1, std), rtol=1e-8), ctx)
                ts = art.get("target_scaler")
                if ts is not None:
                    check("target_scaler_training_only", np.isclose(ts.mean_[0], tr[tc].mean(), rtol=1e-10), ctx)
                if "split" in art:
                    check("artifact_split_matches_protocol", art["split"]["train"] == s["train"], ctx)

                def predict(frame):
                    p = art["model"].predict(sc.transform(frame[f].to_numpy(float)))
                    return ts.inverse_transform(p.reshape(-1, 1)).ravel() if ts is not None else p

                g = d[d.alloy == alloy]
                p = predict(g)
                old = pd.read_csv(W12 / f"results/runs/{key}_predictions.csv", float_precision="round_trip").set_index("row_id")
                check("reproduces_weeks12_predictions", np.allclose(p, old.loc[g.index, "predicted"], rtol=1e-9, atol=1e-7), ctx)
                for rid, pp in zip(g.index, p):
                    pred_rows.append(dict(task=task, alloy=alloy, seed=seed, row_id=int(rid), actual=float(g.at[rid, tc]),
                                          predicted=float(pp), Power=float(g.at[rid, "Power"]), Velocity=float(g.at[rid, "Velocity"])))
                ps = predict(cal)
                for rid, sc_ in zip(cal.index, np.abs(cal[tc].to_numpy(float) - ps)):
                    src_rows.append(dict(task=task, alloy=alloy, seed=seed, row_id=int(rid), score=float(sc_)))
                meta.append(dict(task=task, alloy=alloy, seed=seed, key=key, model_path=str(path.relative_to(W12.parent)),
                                 n_train=len(tr), n_source_calibration=len(cal), train_mean=float(tr[tc].mean())))
                print(key, "checked", flush=True)

(ROOT / "data").mkdir(exist_ok=True)
pd.DataFrame(pred_rows).to_csv(ROOT / "data/ml_predictions.csv", index=False)
pd.DataFrame(src_rows).to_csv(ROOT / "data/source_calibration_scores.csv", index=False)
pd.DataFrame(meta).to_csv(ROOT / "data/model_registry.csv", index=False)
write_json(ROOT / "verification/model_checks.json", dict(passed=len(checks), failed=0, checks=checks))
print("models:", len(meta), "checks passed:", len(checks))
