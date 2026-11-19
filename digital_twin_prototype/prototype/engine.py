"""Raw-input prediction and explicit measurement state; separate from frozen results.

v1.2: geometry outputs use the proposed method of Chapter 4 (shift or linear update chosen by
inner leave-one-out error, jackknife+ range, both bounds projected onto [0, inf)). The method
functions are imported unchanged from method_common.py, a byte-identical copy of the corrected
Chapter 6 code (common.py). The defect class keeps split conformal prediction sets."""

import sys
import json
import math
from copy import deepcopy
from pathlib import Path
import numpy as np
from features import prepare, raw_from_row, ROOT, W12, SPEC
from method_common import choose, jackknife_plus, rank_k

GEOMETRY = {"depth", "width", "length"}
CANDIDATES = ["shift", "linear"]  # proposed method: response is the residual of the frozen model

sys.path.insert(0, str(ROOT.parent / "Thesis_C_Weeks_3_4/prototype"))
from workflow import create_replay, ResearchWorkflow, quantile, digest, LABELS


class InputSession:
    def __init__(self, task, alloy, repeat=0):
        self.replay, self.reveals = create_replay(W12, task, alloy, repeat)
        self.selection = {
            "calibration9": self.replay.context["selection9"],
            "calibration20": self.replay.context["selection20"],
        }
        self.task, self.alloy, self.repeat = task, alloy, repeat
        self.queries = {}
        self.observations = []
        self.history = [{"version": 0, "phase": "source", **quantile(self.replay.source_scores)}]
        # v1.1: record identity by inputs, so a known record cannot be reused under a new query ID.
        self._known = [(rid, rec["features"]) for rid, rec in self.replay.records.items()]

    def _records_with_inputs(self, features):
        return [
            rid
            for rid, f in self._known
            if all(np.isclose(features[k], f[k], rtol=1e-12, atol=1e-10) for k in f)
        ]

    def _cal(self):
        """Target measurements as arrays for the Chapter 6 functions (geometry only)."""
        obs = self.observations
        return {
            "y": np.array([float(o["value"]) for o in obs]),
            "p": np.array([float(o["frozen"]) for o in obs]),
            "X": np.array([[float(o["X"][0]), float(o["X"][1])] for o in obs]).reshape(len(obs), 2),
        }

    def _target_state(self):
        n = len(self.observations)
        if self.task not in GEOMETRY:
            return {
                "version": n,
                "phase": "target_only",
                **quantile([o["score"] for o in self.observations]),
            }
        k = rank_k(n)
        entry = {
            "version": n,
            "phase": "target_update",
            "n_cal": n,
            "k": k,
            "unbounded": bool(k > n),
            "choice": None,
        }
        if k <= n:
            cal = self._cal()
            entry["choice"] = choose(CANDIDATES, cal["p"], cal["X"], cal["y"] - cal["p"])
        return entry

    def state(self):
        return deepcopy(self.history[-1])

    def query(self, query_id, raw, role="prospective"):
        if not isinstance(query_id, str) or not query_id or len(query_id) > 150:
            raise ValueError("A nonempty query identity is required")
        if role not in {"prospective", "fixed_test", "observation_pool"}:
            raise ValueError("Unknown record role")
        features, warnings = prepare(self.task, raw)
        if raw["alloy"] != self.alloy:
            raise ValueError("Inputs must match this session's excluded alloy")
        # Known fixed-test identities cannot be relabelled through the API.
        if query_id in self.replay.fixed_test:
            role = "fixed_test"
        if query_id in self.replay.records:
            expected = self.replay.records[query_id]["features"]
            if not all(
                np.isclose(features[k], expected[k], rtol=1e-12, atol=1e-10) for k in expected
            ):
                raise ValueError("A frozen record identity cannot be attached to changed inputs")
        # v1.1: a new query ID whose inputs equal a known record is bound to that record.
        # If any matching record is a fixed-test record, the query is treated as fixed test.
        if query_id in self.replay.records:
            record_ids = [query_id]
        else:
            record_ids = self._records_with_inputs(features)
            if any(r in self.replay.fixed_test for r in record_ids):
                role = "fixed_test"
        if query_id in self.queries:
            prior = self.queries[query_id]
            if prior["raw"] != raw or prior["role"] != role:
                raise ValueError("Query identities and inputs are immutable")
            prediction = prior["prediction"]
        else:
            prediction = self.replay._prediction({"features": features})
            self.queries[query_id] = {
                "raw": deepcopy(raw),
                "role": role,
                "prediction": deepcopy(prediction),
                "record_ids": list(record_ids),
            }
        artifact = self.replay.artifact
        # Range warning uses source training records only, never test labels.
        from features import FIELDS

        scaler = artifact["scaler"]
        scaled = scaler.transform([[features[k] for k in artifact["features"]]])[0]
        if np.max(abs(scaled)) > 5:
            warnings.append(
                "One or more inputs are over five training standard deviations from the training mean; this is a diagnostic, not a validated confidence estimate."
            )
        state = self.state()
        out = {
            "query_id": query_id,
            "task": self.task,
            "alloy": self.alloy,
            "role": role,
            "state": state,
            "warnings": warnings,
            **deepcopy(prediction),
        }
        if self.task == "classification":
            q = math.inf if state["unbounded"] else state["q"]
            out.update(
                unit="category",
                category_set=[lab for lab in LABELS if 1 - prediction["probabilities"][lab] <= q],
            )
        else:
            p0 = float(prediction["predicted"])
            form = None
            if state["phase"] == "source":
                q = state["q"]
                lo, hi, point = (None, None, p0) if q is None else (p0 - q, p0 + q, p0)
            elif state["unbounded"]:
                lo, hi, point = None, None, p0
            else:
                test = {
                    "p": np.array([p0]),
                    "X": np.array([[float(features["Power"]), float(features["Velocity"])]]),
                }
                l, h, pt, form = jackknife_plus(self._cal(), test, CANDIDATES, "residual")
                lo, hi, point = float(l[0]), float(h[0]), float(pt[0])
            if lo is not None:
                lo, hi = max(lo, 0.0), max(hi, 0.0)  # physical bound of Section 4.3.4
            out.update(
                unit="um",
                frozen_predicted=p0,
                predicted=point,
                update_form=form,
                lower=lo,
                upper=hi,
            )
            if point < 0:
                warnings.append(
                    "The point prediction is negative; it is shown unchanged, and the range is bounded at zero."
                )
            if state["phase"] == "target_update" and state["unbounded"]:
                warnings.append(
                    "Fewer than nine target measurements: the range is unbounded and the prediction is not yet updated."
                )
        return out

    def observe(self, query_id, value, unit):
        if query_id not in self.queries:
            raise ValueError("Record a prediction before adding its measurement")
        if query_id in self.replay.fixed_test or self.queries[query_id]["role"] == "fixed_test":
            raise ValueError("Fixed-test measurements cannot update calibration")
        if query_id in {o["query_id"] for o in self.observations}:
            raise ValueError("Duplicate measurement")
        # v1.1: the same known record cannot enter calibration twice under different query IDs.
        record_ids = set(self.queries[query_id].get("record_ids", []))
        if record_ids & set(self.replay.fixed_test):
            raise ValueError(
                "These inputs belong to a fixed-test record; its measurement cannot update calibration"
            )
        used = (
            set().union(*[set(o.get("record_ids", [])) for o in self.observations])
            if self.observations
            else set()
        )
        if record_ids & used:
            raise ValueError(
                "This known record is already in the calibration set under another query ID"
            )
        p = self.queries[query_id]["prediction"]
        if self.task == "classification":
            if unit != "category" or not isinstance(value, str) or value not in LABELS:
                raise ValueError("Choose a supported defect category")
            score = 1 - p["probabilities"][value]
        else:
            if (
                unit != "um"
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("A finite nonnegative measurement in µm is required")
            score = abs(value - p["predicted"])
        entry = {
            "query_id": query_id,
            "value": value,
            "unit": unit,
            "score": float(score),
            "record_ids": sorted(record_ids),
        }
        if self.task in GEOMETRY:
            features, _ = prepare(self.task, self.queries[query_id]["raw"])
            entry.update(
                frozen=float(p["predicted"]),
                X=[float(features["Power"]), float(features["Velocity"])],
            )
        self.observations.append(entry)
        self.history.append(self._target_state())
        return self.state()

    def reveal_next(self):
        ordered = self.selection["calibration20"] or self.selection["calibration9"]
        used = {o["query_id"] for o in self.observations}
        rid = next((rid for rid in ordered if rid not in used), None)
        if rid is None:
            raise ValueError("All selected demonstration measurements have been used")
        record = self.replay.records[rid]
        result = self.query(rid, raw_from_row(self.task, record["raw_row_id"]), "observation_pool")
        reveal = self.reveals[rid]
        self.observe(rid, reveal["value"], reveal["unit"])
        return {"query": result, "measurement": reveal, "state": self.state()}

    def snapshot(self):
        return {
            "schema_version": 2,
            "task": self.task,
            "alloy": self.alloy,
            "repeat": self.repeat,
            "model_sha256": self.replay.context["model_sha256"],
            "protocol_sha256": self.replay.context["frozen_protocol_sha256"],
            "queries": deepcopy(self.queries),
            "observations": deepcopy(self.observations),
            "history": deepcopy(self.history),
        }

    def save(self, path):
        state = self.snapshot()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"sha256": digest(state), "state": state}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temp.replace(path)

    @classmethod
    def load(cls, path):
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        state = envelope["state"]
        if digest(state) != envelope["sha256"]:
            raise ValueError("Saved state integrity check failed")
        obj = cls(state["task"], state["alloy"], state["repeat"])
        for rid, query in state["queries"].items():
            result = obj.query(rid, query["raw"], query["role"])
            if obj.queries[rid]["prediction"] != query["prediction"]:
                raise ValueError("Saved prediction differs from the model")
        for observation in state["observations"]:
            obj.observe(observation["query_id"], observation["value"], observation["unit"])
        if obj.snapshot() != state:
            raise ValueError("Saved model, scores or history differ from recomputation")
        return obj
