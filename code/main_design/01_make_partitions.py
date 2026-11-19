"""New target-alloy partitions for the confirmatory experiment.

Changes from the 20 Oct partitions, each from an audit finding:
- 50 fresh test draws per alloy (seed base 20261022), not the single fixed test set already inspected.
- Group-aware: rows with identical active model inputs form one condition group;
  a group goes wholly to test or wholly to the pool. This removes the duplicate
  test/pool pairs found in IN625, Ti-6Al-4V, Hastelloy X and others.
- Nested calibration prefixes for budgets 9, 12, 15, 20, 30.
The trained models are unchanged: they never saw any row of the target alloy.
"""
import json
import numpy as np
import pandas as pd
from common import (ROOT, W12, TASKS, BUDGETS, REPEATS, TEST_N, MIN_TEST_N, PRIMARY_MIN_ROWS,
                    SECONDARY_MIN_ROWS, SEED_BASE, write_json)

out = ROOT / "protocol/partitions.json"
if out.exists():
    raise SystemExit("partitions.json exists; refusing to overwrite frozen partitions.")
spec = json.loads((W12 / "protocol/frozen.json").read_text())
parts = []
summary = []
for ti, task in enumerate(TASKS):
    d = pd.read_csv(W12 / f"data/{task}.csv", low_memory=False, float_precision="round_trip").set_index("row_id")
    feats = spec["tasks"][task]["features"]
    alloys = sorted(d.alloy.unique())
    for ai, alloy in enumerate(alloys):
        g = d[d.alloy == alloy]
        if len(g) < SECONDARY_MIN_ROWS:
            continue
        cohort = "primary" if len(g) >= PRIMARY_MIN_ROWS else "secondary"
        key = g[feats].round(9).apply(lambda r: tuple(r), axis=1)
        groups = {}
        for rid, k in zip(g.index, key):
            groups.setdefault(k, []).append(int(rid))
        glist = list(groups.values())
        for r in range(REPEATS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, ti, ai, r]))
            order = rng.permutation(len(glist))
            test = []
            for gi in order:                      # greedy fill to at most TEST_N rows
                if len(test) + len(glist[gi]) <= TEST_N:
                    test += glist[gi]
                if len(test) == TEST_N:
                    break
            test_set = set(test)
            pool = [rid for grp in glist for rid in grp if rid not in test_set]
            cal_order = [int(x) for x in rng.permutation(sorted(pool))]
            # Same budget set in every repeat of an alloy: require the pool left by a full 20-row test set.
            budgets = [b for b in BUDGETS if b <= len(g) - TEST_N and b <= len(pool)]
            if cohort == "secondary":
                budgets = [b for b in budgets if b == 9]
            parts.append(dict(task=task, alloy=alloy, cohort=cohort, repeat=r, n_alloy_rows=len(g),
                              n_condition_groups=len(glist), test_row_ids=sorted(test), pool_row_ids=sorted(pool),
                              calibration_order=cal_order, budgets=budgets))
            assert len(test) >= MIN_TEST_N, (task, alloy, r, len(test))
        summary.append(dict(task=task, alloy=alloy, cohort=cohort, n_rows=len(g), n_condition_groups=len(glist),
                            largest_group=max(map(len, glist))))
write_json(out, parts)
pd.DataFrame(summary).to_csv(ROOT / "protocol/partition_cohorts.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
print("partitions:", len(parts))
