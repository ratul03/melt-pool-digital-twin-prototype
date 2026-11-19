"""Data checks used as evidence in Chapter 3 (written 2 Nov 2026). Reads frozen inputs only."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

W12 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TABLE_A10_K = {
    "SS316L": 1688,
    "Ti-6Al-4V": 1922,
    "IN718": 1673,
    "SS17-4PH": 1693,
    "IN625": 1593,
    "IN738LC": 1543,
    "Hastelloy X": 1628,
    "Cu10Sn": 1272,
    "AlSi10Mg": 1142,
    "Tungsten": 3695,
    "Ti-49Al-2Cr-2Nb": 1795,
    "HCP Cu": 1631,
    "Invar36": 2000,
    "SS304": 1693,
    "WE43": 863,
    "MS1-": 2848,
    "MS1": 2848,
    "CMSX-4": 1653,
    "TiC/Inconel 718": 2015,
    "SS304L": 1693,
    "Ti6242": 1978,
    "4140 steel": 1689,
    "AA7075": 829,
    "Ni-5NB": 1703,
}
PROPS = ["density", "Cp", "k", "melting T"]
facts = {}
rows = []
for task in ["depth", "width", "length", "classification"]:
    d = pd.read_csv(W12 / f"data/{task}.csv", float_precision="round_trip")
    for alloy, g in d.groupby("alloy"):
        r = dict(task=task, alloy=alloy, n=len(g))
        for p in PROPS:
            vals = sorted(g[p].dropna().unique())
            r[p + "_n_values"] = len(vals)
            r[p + "_values"] = "|".join(f"{v:g}" for v in vals)
        ref = TABLE_A10_K.get(alloy)
        r["table_A10_melting_K"] = ref
        r["melting_T_matches_table"] = bool(
            ref is not None and set(g["melting T"].unique()) == {ref}
        )
        rows.append(r)
mp = pd.DataFrame(rows)
mp.to_csv(OUT / "material_properties_per_alloy.csv", index=False)
for task in ["depth", "width", "length", "classification"]:
    z = mp[mp.task == task]
    facts[f"{task}_alloys_with_more_than_one_property_value"] = z.loc[
        (z[[p + "_n_values" for p in PROPS]] > 1).any(axis=1), "alloy"
    ].tolist()
    facts[f"{task}_alloys_melting_T_not_equal_table_A10"] = z.loc[
        ~z.melting_T_matches_table, "alloy"
    ].tolist()

# Source publications for the tested (>=20-record) alloys.
counts = pd.read_csv(W12 / "data/alloy_counts.csv")
ov = pd.read_csv(W12 / "data/alloy_source_overlap.csv")
src = counts.merge(
    ov[["task", "alloy", "sources_shared_with_training", "test_rows_from_shared_sources"]],
    on=["task", "alloy"],
)
src = src[src.n >= 20][
    [
        "task",
        "alloy",
        "n",
        "sources",
        "sources_shared_with_training",
        "test_rows_from_shared_sources",
    ]
]
src.to_csv(OUT / "source_publications_tested_alloys.csv", index=False)
for task, z in src.groupby("task"):
    facts[f"{task}_tested_alloys"] = int(len(z))
    facts[f"{task}_publications_per_tested_alloy_min_max"] = [
        int(z.sources.min()),
        int(z.sources.max()),
    ]
    facts[f"{task}_max_publications_shared_with_other_alloys"] = int(
        z.sources_shared_with_training.max()
    )

# Fitting-set sizes: random split vs leave-one-alloy-out, per seed.
splits = json.loads((W12 / "protocol/splits.json").read_text())
fs = []
for s in splits:
    if s["protocol"] == "leave_one_alloy_out" and not s["primary"]:
        continue
    fs.append(
        dict(
            task=s["task"],
            protocol=s["protocol"],
            seed=s["seed"],
            alloy=s["alloy"],
            n_fit=len(s["train"]),
            n_calibration=len(s["calibration"]),
            n_test=len(s["test"]),
        )
    )
fs = pd.DataFrame(fs)
fs.to_csv(OUT / "fitting_set_sizes.csv", index=False)
for task, z in fs.groupby("task"):
    larger = []
    for seed, zz in z.groupby("seed"):
        rnd = zz.loc[zz.protocol == "random", "n_fit"].iloc[0]
        lo = zz[zz.protocol == "leave_one_alloy_out"]
        larger.append((int((lo.n_fit > rnd).sum()), len(lo), int(rnd)))
    facts[f"{task}_loao_fitting_sets_larger_than_random_per_seed"] = larger
(OUT / "data_check_facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
