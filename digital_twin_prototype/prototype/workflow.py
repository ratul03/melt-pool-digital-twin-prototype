"""Versioned, fixed-model uncertainty replay over the frozen research records.

The engine stores feature-only target records. Labels enter only via observe(),
and only for the frozen permitted observation pool. Fixed-test labels are never
loaded into this engine. This is a bounded research workflow, not a security
boundary or a production data collection system.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LABELS = ("LOF", "balling", "desirable", "keyhole")
FEATURE_UNITS = "inherited-preprocessed-feature-values"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def quantile(scores):
    """Exact finite-sample 90% rank, without interpolation or clipping."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Calibration scores must be finite, nonnegative scalars")
    n = len(scores)
    k = (9 * (n + 1) + 9) // 10
    # None is persisted rather than the nonstandard JSON Infinity token.
    return {
        "n_cal": n,
        "k": k,
        "q": None if k > n else float(np.sort(scores)[k - 1]),
        "unbounded": bool(k > n),
    }


def record_id(task, raw_row_id):
    dataset = "classification" if task == "classification" else "regression"
    return f"meltpoolnet:{dataset}:{task}:{int(raw_row_id)}"


class ResearchWorkflow:
    def __init__(self, model_path, context, records, source_scores):
        self.model_path = Path(model_path)
        self.context = deepcopy(context)
        if sha256(self.model_path) != context["model_sha256"]:
            raise ValueError("Saved-model hash mismatch")
        self.artifact = joblib.load(self.model_path)
        self.task = context["task"]
        self.alloy = context["alloy"]
        if self.task not in {"depth", "width", "length", "classification"}:
            raise ValueError("Unknown task")
        if self.artifact["protocol_sha256"] != context["frozen_protocol_sha256"]:
            raise ValueError("Model was not trained under the pinned protocol")
        split = self.artifact["split"]
        if split["task"] != self.task or split["alloy"] != self.alloy or split["seed"] != 0:
            raise ValueError("Model/task/alloy mismatch")
        if digest(split) != context["model_split_digest"]:
            raise ValueError("Saved-model split differs from selected split")
        self.allowed = set(context["allowed_observation_ids"])
        self.fixed_test = set(context["fixed_test_ids"])
        if self.allowed & self.fixed_test:
            raise ValueError("Observation/test leakage")
        self.records = deepcopy(records)
        if (
            digest(records) != context["records_digest"]
            or digest([float(s) for s in source_scores]) != context["source_scores_digest"]
        ):
            raise ValueError("Record/source-score content differs from pinned replay inputs")
        if set(self.records) != self.allowed | self.fixed_test:
            raise ValueError("Record registry does not match frozen target roles")
        for rid, record in self.records.items():
            if rid != record["record_id"]:
                raise ValueError("Record identity mismatch")
            self.validate_record(record, bind_registry=False)
        self.source_scores = [float(s) for s in source_scores]
        if len(self.source_scores) != len(context["source_calibration_ids"]):
            raise ValueError("Source score count differs from pinned calibration records")
        if self.task == "classification" and any(s > 1 for s in self.source_scores):
            raise ValueError("Probability scores must be in [0,1]")
        self.predictions = {}
        self.observations = []
        self.history = [
            {"version": 0, "phase": "source", **quantile(self.source_scores), "observation_ids": []}
        ]

    def validate_record(self, record, bind_registry=True):
        if set(record) != {"record_id", "raw_row_id", "task", "alloy", "feature_units", "features"}:
            raise ValueError(
                "Record requires identity, task, alloy, feature units and features only"
            )
        if record["task"] != self.task or record["alloy"] != self.alloy:
            raise ValueError("Wrong task or alloy")
        rid = record["record_id"]
        if rid != record_id(self.task, record["raw_row_id"]):
            raise ValueError("Record must include dataset and task in its identity")
        if rid not in self.allowed | self.fixed_test:
            raise ValueError("Unknown record")
        if record["feature_units"] != FEATURE_UNITS:
            raise ValueError("Features must use the inherited prepared-data schema")
        features = record["features"]
        if set(features) != set(self.artifact["features"]):
            raise ValueError("Missing or unexpected feature")
        for name, value in features.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid feature {name}")
        # A replay record is immutable: its ID cannot be attached to new inputs.
        if bind_registry and record != self.records[rid]:
            raise ValueError("Inputs differ from the frozen record")

    def _prediction(self, record):
        x = np.array([[record["features"][f] for f in self.artifact["features"]]], dtype=float)
        x = self.artifact["scaler"].transform(x)
        model = self.artifact["model"]
        predicted = model.predict(x)[0]
        if self.task == "classification":
            probs = model.predict_proba(x)[0]
            mapped = {
                label: (
                    float(probs[list(model.classes_).index(label)])
                    if label in model.classes_
                    else 0.0
                )
                for label in LABELS
            }
            if not np.isclose(sum(mapped.values()), 1) or any(
                not 0 <= v <= 1 for v in mapped.values()
            ):
                raise ValueError("Invalid classifier probabilities")
            return {"predicted": str(predicted), "probabilities": mapped}
        if self.artifact["target_scaler"] is not None:
            predicted = self.artifact["target_scaler"].inverse_transform([[predicted]])[0, 0]
        if not np.isfinite(predicted):
            raise ValueError("Nonfinite model prediction")
        return {"predicted": float(predicted)}

    def query(self, record):
        self.validate_record(record)
        rid = record["record_id"]
        if rid not in self.predictions:
            self.predictions[rid] = self._prediction(record)
        prediction = deepcopy(self.predictions[rid])
        state = self.history[-1]
        result = {
            "record_id": rid,
            "task": self.task,
            "alloy": self.alloy,
            "calibration_version": state["version"],
            "calibration_phase": state["phase"],
            "n_cal": state["n_cal"],
            "q": state["q"],
            "unbounded": state["unbounded"],
            **prediction,
        }
        if self.task == "classification":
            q = math.inf if state["unbounded"] else state["q"]
            result.update(
                unit="category",
                category_set=[lab for lab in LABELS if 1 - prediction["probabilities"][lab] <= q],
            )
        else:
            q = state["q"]
            result.update(
                unit="um",
                lower=None if q is None else prediction["predicted"] - q,
                upper=None if q is None else prediction["predicted"] + q,
            )
        return result

    def observe(self, *, record_id, task, alloy, value, unit):
        if task != self.task or alloy != self.alloy:
            raise ValueError("Observation belongs to another task/alloy")
        if record_id in self.fixed_test:
            raise ValueError("Fixed-test labels may never update calibration")
        if record_id not in self.allowed:
            raise ValueError("Observation not in frozen observation pool")
        if record_id in {row["record_id"] for row in self.observations}:
            raise ValueError("Duplicate observation")
        if record_id not in self.predictions:
            raise ValueError("Prediction must be recorded before revealing its measurement")
        prediction = self.predictions[record_id]
        if self.task == "classification":
            if unit != "category" or not isinstance(value, str) or value not in LABELS:
                raise ValueError(
                    "Classification observation requires one of the fixed four categories"
                )
            score = 1 - prediction["probabilities"][value]
        else:
            if (
                unit != "um"
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Geometry measurement must be finite, nonnegative and in um")
            value = float(value)
            score = abs(value - prediction["predicted"])
        self.observations.append(
            {
                "record_id": record_id,
                "task": task,
                "alloy": alloy,
                "value": value,
                "unit": unit,
                "score": float(score),
            }
        )
        # Source observations are replaced, not pooled with the target observations.
        self.history.append(
            {
                "version": len(self.observations),
                "phase": "target_only",
                **quantile([o["score"] for o in self.observations]),
                "observation_ids": [o["record_id"] for o in self.observations],
            }
        )
        return deepcopy(self.history[-1])

    def snapshot(self):
        return {
            "schema_version": 1,
            "context": deepcopy(self.context),
            "records": deepcopy(self.records),
            "source_scores": list(self.source_scores),
            "predictions": deepcopy(self.predictions),
            "observations": deepcopy(self.observations),
            "history": deepcopy(self.history),
        }

    def save(self, path):
        state = self.snapshot()
        envelope = {"state_sha256": digest(state), "state": state}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(envelope, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, model_path, expected_context):
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        state = envelope["state"]
        if digest(state) != envelope["state_sha256"]:
            raise ValueError("Persisted state integrity mismatch")
        if state["context"] != expected_context:
            raise ValueError("Persisted state no longer matches pinned context")
        obj = cls(model_path, state["context"], state["records"], state["source_scores"])
        # Recompute cached model outputs and each history entry instead of trusting the file.
        for rid, prediction in state["predictions"].items():
            obj.query(obj.records[rid])
            if obj.predictions[rid] != prediction:
                raise ValueError("Persisted prediction disagrees with saved model")
        for observation in state["observations"]:
            obj.observe(
                **{key: observation[key] for key in ["record_id", "task", "alloy", "value", "unit"]}
            )
            if obj.observations[-1] != observation:
                raise ValueError("Persisted observation score differs from recomputation")
        if obj.snapshot() != state:
            raise ValueError("Persisted history differs from replay")
        return obj


def create_replay(weeks12, task, alloy, repeat=0):
    """Build feature-only registry; return permitted reveal labels separately.

    CSV targets on fixed-test rows are dropped before records enter the engine.
    Evaluation of test outcomes belongs to the offline experiment, not this API.
    """
    weeks12 = Path(weeks12)
    frozen_path = weeks12 / "protocol/frozen.json"
    frozen = json.loads(frozen_path.read_text())
    for rel, expected in frozen["input_hashes"].items():
        if sha256(weeks12 / rel) != expected:
            raise ValueError(f"Inherited input hash mismatch: {rel}")
    future_path = weeks12 / "protocol/future_calibration_splits.json"
    future = next(
        s
        for s in json.loads(future_path.read_text())
        if s["task"] == task and s["alloy"] == alloy and s["repeat"] == repeat
    )
    parts = json.loads((weeks12 / "protocol/splits.json").read_text())
    index, split = next(
        (i, s)
        for i, s in enumerate(parts)
        if s["task"] == task
        and s["alloy"] == alloy
        and s["protocol"] == "leave_one_alloy_out"
        and s["seed"] == 0
    )
    key = f"{task}_leave_one_alloy_out_{index:03d}_s0"
    model_path = weeks12 / "models" / f"{key}.joblib"
    art = joblib.load(model_path)
    data = pd.read_csv(weeks12 / "data" / f"{task}.csv").set_index("row_id")
    cfg = frozen["tasks"][task]
    target_ids = set(future["pool_row_ids"]) | set(future["test_row_ids"])
    if set(split["train"]) & target_ids or set(split["calibration"]) & target_ids:
        raise ValueError("Target-alloy records entered fitting/source calibration")
    if set(data.loc[split["calibration"], "alloy"]) & {alloy}:
        raise ValueError("Target alloy entered source calibration")
    if not target_ids <= set(split["test"]) or set(data.loc[sorted(target_ids), "alloy"]) != {
        alloy
    }:
        raise ValueError("Future target records disagree with model split")
    source = data.loc[split["calibration"]]
    x = art["scaler"].transform(source[art["features"]].to_numpy())
    if task == "classification":
        probs = art["model"].predict_proba(x)
        classes = list(art["model"].classes_)
        scores = [
            1 - float(probs[i, classes.index(label)]) if label in classes else 1.0
            for i, label in enumerate(source[cfg["target"]])
        ]
    else:
        p = art["model"].predict(x)
        if art["target_scaler"] is not None:
            p = art["target_scaler"].inverse_transform(p.reshape(-1, 1)).ravel()
        scores = np.abs(source[cfg["target"]].to_numpy() - p).tolist()
    records = {}
    for row_id in sorted(target_ids):
        rid = record_id(task, row_id)
        records[rid] = {
            "record_id": rid,
            "raw_row_id": row_id,
            "task": task,
            "alloy": alloy,
            "feature_units": FEATURE_UNITS,
            "features": {f: float(data.at[row_id, f]) for f in cfg["features"]},
        }
    context = {
        "task": task,
        "alloy": alloy,
        "repeat": repeat,
        "model_seed": 0,
        "key": key,
        "model_sha256": sha256(model_path),
        "frozen_protocol_sha256": sha256(frozen_path),
        "future_splits_sha256": sha256(future_path),
        "model_split_digest": digest(split),
        "data_sha256": sha256(weeks12 / "data" / f"{task}.csv"),
        "allowed_observation_ids": [record_id(task, i) for i in future["pool_row_ids"]],
        "fixed_test_ids": [record_id(task, i) for i in future["test_row_ids"]],
        "source_calibration_ids": [record_id(task, i) for i in split["calibration"]],
        "selection9": [record_id(task, i) for i in future["calibration9"]],
        "selection20": (
            None
            if future["calibration20"] is None
            else [record_id(task, i) for i in future["calibration20"]]
        ),
        "records_digest": digest(records),
        "source_scores_digest": digest([float(s) for s in scores]),
        "research_only": True,
        "live_machine_connection": False,
    }
    reveals = {}
    for i in future["pool_row_ids"]:
        value = data.at[i, cfg["target"]]
        reveals[record_id(task, i)] = {
            "record_id": record_id(task, i),
            "task": task,
            "alloy": alloy,
            "value": str(value) if task == "classification" else float(value),
            "unit": "category" if task == "classification" else "um",
        }
    return ResearchWorkflow(model_path, context, records, scores), reveals
