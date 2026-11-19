"""v1.2 verification: raw features, all 22 replay scenarios, persistence and invalid-input tests.

Geometry ranges after target measurements are checked against (1) jackknife_plus from the
corrected Chapter 6 code and (2) an independent brute-force jackknife+ written here.
Usage: python verify_prototype_checks.py [task ...]   (default: all tasks)"""

import sys
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from engine import InputSession
from features import ROOT, OUT, W12, SPEC, FIELDS, prepare, raw_from_row
from method_common import jackknife_plus

TASKS_RUN = sys.argv[1:] or ["depth", "width", "length", "classification"]
C22 = OUT.parent / "Thesis results C 22-10-2026 corrected 2026-11-02"
CH6_COMMON_SHA256 = "aab1d503fd080415040dc789fa3752fb077dd851be2957f2d46169ccf3524633"


def jk_brute(cal, test):
    """Independent jackknife+ (explicit refits, no hat matrix) for the shift/linear residual update."""
    y, p = cal["y"], cal["p"]
    t = y - p
    n = len(y)
    pt = test["p"]

    def design(form, pp):
        return np.ones((len(pp), 1)) if form == "shift" else np.column_stack([np.ones(len(pp)), pp])

    def fit(form, pp, tt):
        return np.linalg.lstsq(design(form, pp), tt, rcond=None)[0]

    def loo_mse(form, pp, tt):
        e = []
        for j in range(len(tt)):
            m = np.arange(len(tt)) != j
            e.append(tt[j] - (design(form, pp[j : j + 1]) @ fit(form, pp[m], tt[m]))[0])
        return float(np.mean(np.square(e)))

    def pick(pp, tt):
        return "shift" if loo_mse("shift", pp, tt) <= loo_mse("linear", pp, tt) else "linear"

    k = -(-9 * (n + 1) // 10)
    lo, hi = [], []
    for i in range(n):
        m = np.arange(n) != i
        c = pick(p[m], t[m])
        b = fit(c, p[m], t[m])
        R = abs(y[i] - ((design(c, p[i : i + 1]) @ b)[0] + p[i]))
        mu = (design(c, pt) @ b)[0] + pt[0]
        lo.append(mu - R)
        hi.append(mu + R)
    c = pick(p, t)
    b = fit(c, p, t)
    return sorted(lo)[n - k], sorted(hi)[k - 1], (design(c, pt) @ b)[0] + pt[0], c


CHECKS = []
DEMOS = []


def check(name, condition, context=""):
    CHECKS.append({"name": name, "passed": bool(condition), "context": str(context)})


def rejects(name, call):
    try:
        call()
    except (ValueError, KeyError, TypeError):
        check(name, True)
    else:
        check(name, False)


def close(a, b):
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-7))


def main():
    output = OUT / "verification/prototype_states"
    output.mkdir(parents=True, exist_ok=True)
    rejected = []
    check(
        "method_code_identical_to_chapter6",
        hashlib.sha256((Path(__file__).parent / "method_common.py").read_bytes()).hexdigest()
        == CH6_COMMON_SHA256,
    )
    ml22 = pd.read_csv(C22 / "data/ml_predictions.csv", float_precision="round_trip")
    ml22 = ml22[ml22.seed == 0].set_index(["task", "alloy", "row_id"])["predicted"]
    for task, cfg in SPEC["tasks"].items():
        if task not in TASKS_RUN:
            continue
        data = pd.read_csv(W12 / f"data/{task}.csv", float_precision="round_trip").set_index(
            "row_id"
        )
        for rid, row in data.iterrows():
            try:
                features, warnings = prepare(task, raw_from_row(task, rid))
                check(
                    "raw_features_equal_frozen_features",
                    close(
                        [features[f] for f in cfg["features"]], row[cfg["features"]].to_numpy(float)
                    ),
                    (task, rid),
                )
            except ValueError as e:
                rejected.append({"task": task, "row_id": int(rid), "reason": str(e)})
        print(task, "raw inputs checked", len(data), flush=True)
    pd.DataFrame(rejected, columns=["task", "row_id", "reason"]).to_csv(
        OUT / f"verification/raw_input_rejections_{'_'.join(TASKS_RUN)}.csv", index=False
    )
    check("all_frozen_records_supported", not rejected, rejected[:10])
    offline = pd.read_csv(
        ROOT.parent / "Thesis_C_Weeks_3_4/results/update_predictions.csv",
        low_memory=False,
        float_precision="round_trip",
    )
    offline = offline[(offline.model_seed == 0) & (offline.repeat == 0)]
    future = [
        f
        for f in json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
        if f["repeat"] == 0
    ]
    for f in future:
        if f["task"] not in TASKS_RUN:
            continue
        session = InputSession(f["task"], f["alloy"])
        task, alloy = f["task"], f["alloy"]
        demo = {"task": task, "alloy": alloy, "budgets": []}

        def compare(budget):
            points = []
            for rid in session.replay.context["fixed_test_ids"]:
                rawid = session.replay.records[rid]["raw_row_id"]
                result = session.query(rid, raw_from_row(task, rawid), "fixed_test")
                row = offline[
                    (offline.task == task)
                    & (offline.alloy == alloy)
                    & (offline.budget == budget)
                    & (offline.row_id == rawid)
                ]
                check("exact_offline_match_exists", len(row) == 1, (task, alloy, budget, rid))
                expected = row.iloc[0]
                if task == "classification" or budget == 0:
                    check(
                        "offline_calibration_margin",
                        close(result["state"]["q"], expected.q),
                        (task, alloy, budget),
                    )
                if task == "classification":
                    check(
                        "offline_class_and_set",
                        result["predicted"] == expected.predicted
                        and result["category_set"] == json.loads(expected.set_json),
                        (task, alloy, budget, rid),
                    )
                    check(
                        "offline_class_probabilities",
                        close(
                            list(result["probabilities"].values()),
                            [expected["prob_" + lab] for lab in result["probabilities"]],
                        ),
                        rid,
                    )
                else:
                    check(
                        "frozen_prediction_matches_offline",
                        close(result["frozen_predicted"], expected.predicted),
                        (task, alloy, budget, rid),
                    )
                    key = (task, alloy, int(rawid))
                    if key in ml22.index:
                        check(
                            "frozen_prediction_matches_chapter6_cache",
                            close(result["frozen_predicted"], ml22.loc[key]),
                            key,
                        )
                    if budget == 0:
                        check(
                            "source_range_matches_offline_with_bound",
                            close(
                                [result["lower"], result["upper"]],
                                [max(expected.lower, 0.0), max(expected.upper, 0.0)],
                            ),
                            (task, alloy, rid),
                        )
                    else:
                        feats, _ = prepare(task, raw_from_row(task, rawid))
                        test = {
                            "p": np.array([result["frozen_predicted"]]),
                            "X": np.array([[feats["Power"], feats["Velocity"]]]),
                        }
                        cal = session._cal()
                        l, h, pt, c = jackknife_plus(cal, test, ["shift", "linear"], "residual")
                        check(
                            "update_matches_chapter6_code",
                            close(
                                [result["predicted"], result["lower"], result["upper"]],
                                [pt[0], max(l[0], 0.0), max(h[0], 0.0)],
                            )
                            and result["update_form"] == c,
                            (task, alloy, budget, rid),
                        )
                        bl, bh, bp, bc = jk_brute(cal, test)
                        check(
                            "update_matches_bruteforce",
                            close(
                                [result["predicted"], result["lower"], result["upper"]],
                                [bp, max(bl, 0.0), max(bh, 0.0)],
                            )
                            and result["update_form"] == bc,
                            (task, alloy, budget, rid),
                        )
                        check(
                            "update_range_ordered_nonnegative",
                            0.0 <= result["lower"] <= result["upper"],
                            (task, alloy, budget, rid),
                        )
                        check(
                            "state_choice_matches_query",
                            session.state()["choice"] == result["update_form"],
                            (task, alloy, budget),
                        )
                points.append(
                    result["frozen_predicted"] if task != "classification" else result["predicted"]
                )
            demo["budgets"].append(
                {
                    "budget": budget,
                    "state": session.state(),
                    "example": result,
                    "n_test_queries": len(points),
                }
            )
            return points

        initial = compare(0)
        maximum = 20 if f["primary_budget_comparison"] else 9
        for n in range(1, maximum + 1):
            session.reveal_next()
            check(
                "finite_sample_unbounded_below_nine",
                session.state()["unbounded"] == (n < 9),
                (task, alloy, n),
            )
            if n in {9, 20}:
                after = compare(n)
                check("fixed_point_predictions", after == initial, (task, alloy, n))
        p = output / f"{task}_{len(DEMOS):02d}.json"
        session.save(p)
        restored = InputSession.load(p)
        check("save_reload_exact_state", restored.snapshot() == session.snapshot(), (task, alloy))
        testid = session.replay.context["fixed_test_ids"][0]
        rejects(
            "fixed_test_observation_rejected",
            lambda: session.observe(
                testid,
                "LOF" if task == "classification" else 100,
                "category" if task == "classification" else "um",
            ),
        )
        observation = session.observations[0]
        rejects(
            "duplicate_observation_rejected",
            lambda: session.observe(
                observation["query_id"], observation["value"], observation["unit"]
            ),
        )
        rejects(
            "observation_without_prediction_rejected",
            lambda: session.observe("never-predicted", 100, "um"),
        )
        rawid = session.replay.records[testid]["raw_row_id"]
        raw = raw_from_row(task, rawid)
        wrong = dict(raw, power_W=-1)
        rejects("negative_input_rejected", lambda: session.query("bad-power", wrong))
        wrong = dict(raw, power_W=float("nan"))
        rejects("nonfinite_input_rejected", lambda: session.query("bad-nan", wrong))
        wrong = dict(raw, process="DED")
        rejects("unsupported_process_rejected", lambda: session.query("bad-process", wrong))
        wrong = dict(raw, alloy="other")
        rejects("wrong_alloy_rejected", lambda: session.query("bad-alloy", wrong))
        wrong = dict(raw)
        wrong.pop("power_W")
        rejects("missing_input_rejected", lambda: session.query("bad-missing", wrong))
        wrong = dict(raw, measured_width_um=100)
        rejects("postbuild_input_rejected", lambda: session.query("bad-extra", wrong))
        tampered = json.loads(p.read_text())
        if "q" in tampered["state"]["history"][-1]:
            tampered["state"]["history"][-1]["q"] += 0.125
        else:
            tampered["state"]["observations"][0][
                "value"
            ] += 0.125  # geometry v1.2: change one measured value
        corrupt = p.with_suffix(".corrupt.json")
        corrupt.write_text(json.dumps(tampered), encoding="utf-8")
        rejects("corrupted_state_rejected", lambda: InputSession.load(corrupt))
        # Explicit manual trial, using permitted inputs, cannot enter frozen result files.
        manual = dict(raw)
        manual["power_W"] *= 1.001
        session.query("manual-test", manual)
        rejects(
            "wrong_measurement_unit_rejected", lambda: session.observe("manual-test", 100, "mm")
        )
        session.observe(
            "manual-test",
            "desirable" if task == "classification" else 100.0,
            "category" if task == "classification" else "um",
        )
        check(
            "manual_measurement_version_increment",
            session.state()["version"] == maximum + 1,
            (task, alloy),
        )
        DEMOS.append(demo)
        print(task, alloy, "replay, raw inputs and persistence checked", flush=True)
    fail = [c for c in CHECKS if not c["passed"]]
    result = {
        "passed": len(CHECKS) - len(fail),
        "failed": len(fail),
        "failures": fail,
        "scenarios": len(DEMOS),
        "raw_input_rejections": len(rejected),
        "check_counts": pd.Series([c["name"] for c in CHECKS]).value_counts().to_dict(),
    }
    tag = "_".join(TASKS_RUN)
    (OUT / f"verification/prototype_checks_{tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (OUT / f"verification/prototype_demonstrations_{tag}.json").write_text(
        json.dumps(DEMOS, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
