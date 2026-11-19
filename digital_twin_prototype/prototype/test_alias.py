"""v1.1 alias tests: a known record cannot be reused under a new query ID."""

import json
import random
from features import OUT, W12, raw_from_row
from engine import InputSession

checks = []


def check(name, ok, ctx=""):
    checks.append(dict(check=name, passed=bool(ok), context=str(ctx)))


def refused(call):
    try:
        call()
        return False
    except ValueError:
        return True


future = json.loads((W12 / "protocol/future_calibration_splits.json").read_text())
pairs = sorted({(f["task"], f["alloy"]) for f in future if f["repeat"] == 0})
for task, alloy in pairs:
    s = InputSession(task, alloy, 0)
    unit = "category" if task == "classification" else "um"
    # 1. Fixed-test record submitted under new IDs: forced to fixed_test, measurement refused.
    for i, rid in enumerate(sorted(s.replay.fixed_test)[:3]):
        rec = s.replay.records[rid]
        raw = raw_from_row(task, rec["raw_row_id"])
        out = s.query(f"alias-test-{i}", raw)
        check("alias_of_fixed_test_is_fixed_test", out["role"] == "fixed_test", (task, alloy, rid))
        value = (
            s.reveals[rid]["value"]
            if rid in s.reveals
            else ("desirable" if task == "classification" else 100.0)
        )
        check(
            "alias_of_fixed_test_measurement_refused",
            refused(lambda: s.observe(f"alias-test-{i}", value, unit)),
            (task, alloy, rid),
        )
    n_before = len(s.observations)
    check("no_calibration_from_fixed_test_aliases", n_before == 0, (task, alloy))
    # 2. Pool record revealed, then the same record under a new ID: second measurement refused.
    r = s.reveal_next()
    rid = r["query"]["query_id"]
    raw = raw_from_row(task, s.replay.records[rid]["raw_row_id"])
    s.query("alias-pool", raw)
    check(
        "alias_of_used_record_refused",
        refused(lambda: s.observe("alias-pool", r["measurement"]["value"], unit)),
        (task, alloy, rid),
    )
    check("calibration_count_unchanged_after_alias", len(s.observations) == 1, (task, alloy))
    # 3. A genuinely new input is still accepted.
    new = dict(raw)
    key = "power_W"
    new[key] = float(new[key]) * 1.013
    s.query("new-experiment", new)
    s.observe("new-experiment", "desirable" if task == "classification" else 123.4, unit)
    check("new_inputs_accepted", len(s.observations) == 2, (task, alloy))
    # 4. Save and reload keep the record bindings.
    path = OUT / "verification/alias_states" / f"{task}_{alloy.replace(' ', '_')}.json"
    s.save(path)
    t = InputSession.load(path)
    check("reload_keeps_bindings", t.snapshot() == s.snapshot(), (task, alloy))
res = dict(
    passed=sum(c["passed"] for c in checks),
    failed=sum(not c["passed"] for c in checks),
    sessions=len(pairs),
    failures=[c for c in checks if not c["passed"]][:20],
)
(OUT / "verification/alias_checks.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("passed", "failed", "sessions")}), res["failures"][:3])
