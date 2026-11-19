from pathlib import Path
import sys
import json
import time
import warnings
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inputs/code"))
import reproduce as b

spec = json.loads((ROOT / "protocol/frozen.json").read_text())
for rel, h in spec["input_hashes"].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h, rel
parts = json.loads((ROOT / "protocol/splits.json").read_text())
data = {t: pd.read_csv(ROOT / "data" / f"{t}.csv").set_index("row_id") for t in spec["tasks"]}
out = ROOT / "results/runs"
out.mkdir(exist_ok=True)
LABELS = b.LABELS


def metrics(y, p, classification):
    if classification:
        return dict(
            accuracy=float(accuracy_score(y, p)),
            macro_f1=float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
            balanced_accuracy=float(
                recall_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)
            ),
        )
    return dict(
        R2=float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None,
        MAE=float(mean_absolute_error(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        negative_predictions=int(np.sum(p < 0)),
    )


with threadpool_limits(limits=1):
    for k, s in enumerate(parts):
        key = f"{s['task']}_{s['protocol']}_{k:03d}_s{s['seed']}"
        if (out / f"{key}.json").exists():
            continue
        t = s["task"]
        c = t == "classification"
        cfg = spec["tasks"][t]
        d = data[t]
        f = cfg["features"]
        tc = cfg["target"]
        tr, cal, te = [d.loc[s[x]] for x in ["train", "calibration", "test"]]
        assert (
            not set(tr.index) & set(te.index)
            and not set(cal.index) & set(te.index)
            and not set(tr.index) & set(cal.index)
        )
        if s["protocol"] == "leave_one_alloy_out":
            assert set(te.alloy).isdisjoint(set(tr.alloy) | set(cal.alloy))
        xsc = StandardScaler().fit(tr[f].to_numpy())
        ysc = None
        y = tr[tc].to_numpy()
        if cfg["target_scaling"]:
            ysc = StandardScaler().fit(y.reshape(-1, 1))
            y = ysc.transform(y.reshape(-1, 1)).ravel()
        model = b.estimator(cfg["model"], t, s["seed"], "corrected")
        start = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(xsc.transform(tr[f].to_numpy()), y)
        p = model.predict(xsc.transform(te[f].to_numpy()))
        if ysc is not None:
            p = ysc.inverse_transform(p.reshape(-1, 1)).ravel()
        actual = te[tc].to_numpy()
        dummy = np.repeat(tr[tc].mode().iloc[0] if c else tr[tc].mean(), len(te))
        row = {
            **{x: s[x] for x in ["task", "protocol", "seed", "alloy", "primary"]},
            "key": key,
            "n_train": len(tr),
            "n_calibration": len(cal),
            "n_test": len(te),
            "seconds": time.time() - start,
            "warnings": [str(x.message) for x in w],
            "iterations": int(getattr(model, "n_iter_", 0)),
            "metrics": metrics(actual, p, c),
            "dummy_metrics": metrics(actual, dummy, c),
            "scaler_mean": xsc.mean_.tolist(),
            "scaler_scale": xsc.scale_.tolist(),
            "target_scaler_mean": None if ysc is None else float(ysc.mean_[0]),
            "missing_train_classes": sorted(set(LABELS) - set(tr[tc])) if c else [],
            "test_classes_absent_in_train": sorted(set(actual) - set(tr[tc])) if c else [],
        }
        frame = pd.DataFrame(
            dict(
                row_id=te.index,
                alloy=te.alloy.to_numpy(),
                source=te.source.to_numpy(),
                actual=actual,
                predicted=p,
                dummy=dummy,
            )
        )
        if c:
            probs = model.predict_proba(xsc.transform(te[f].to_numpy()))
            for lab in LABELS:
                frame["prob_" + lab] = (
                    probs[:, list(model.classes_).index(lab)] if lab in model.classes_ else 0.0
                )
            row["confusion_matrix"] = confusion_matrix(actual, p, labels=LABELS).tolist()
        frame.to_csv(out / f"{key}_predictions.csv", index=False)
        if s["seed"] == 0:
            art = dict(
                model=model,
                scaler=xsc,
                target_scaler=ysc,
                features=f,
                target=tc,
                split=s,
                protocol_sha256=hashlib.sha256(
                    (ROOT / "protocol/frozen.json").read_bytes()
                ).hexdigest(),
                research_only=True,
            )
            dest = ROOT / "models" / f"{key}.joblib"
            joblib.dump(art, dest, compress=3)
            re = joblib.load(dest)
            p2 = re["model"].predict(re["scaler"].transform(te[f].to_numpy()))
            if ysc is not None:
                p2 = re["target_scaler"].inverse_transform(p2.reshape(-1, 1)).ravel()
            assert np.array_equal(p, p2)
            row["saved_model"] = str(dest.relative_to(ROOT))
            row["reload_pass"] = True
        (out / f"{key}.json").write_text(
            json.dumps(row, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            k + 1,
            "/",
            len(parts),
            key,
            "N",
            len(te),
            row["metrics"],
            "seconds",
            round(row["seconds"], 2),
            flush=True,
        )
print("BASELINE FITS COMPLETE", flush=True)
