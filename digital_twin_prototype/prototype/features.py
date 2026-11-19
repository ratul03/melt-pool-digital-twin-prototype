"""Physical-input adapter for the frozen task-specific models."""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import os

# v1.1: the frozen 20 Oct release is read from its own folder; nothing is written there.
ROOT = Path(
    os.environ.get(
        "THESIS_RELEASE_ROOT",
        Path(__file__).resolve().parents[2]
        / "Thesis_C_Results_2026-10-20"
        / "Thesis_C_Final_Results",
    )
)
OUT = Path(__file__).resolve().parents[1]
W12 = ROOT.parent / "Thesis_C_Weeks_1_2"
SPEC = json.loads((W12 / "protocol/frozen.json").read_text())
CONSTANTS = json.loads((W12 / "inputs/code/reference_constants.json").read_text())
REG = pd.read_csv(W12 / "inputs/data/raw/meltpoolnet_regression.csv", low_memory=False)
CLS = pd.read_csv(W12 / "inputs/data/raw/meltpoolnet_classification.csv", low_memory=False)
ETA = {
    str(m).strip(): float(g.dropna().iloc[0])
    for m, g in REG.groupby("Material")["minimum absorptivity"]
    if len(g.dropna()) and g.dropna().nunique() == 1
}
FIELDS = {
    "power_W": ("Power", "Laser power", "W"),
    "speed_mm_s": ("Velocity", "Scanning speed", "mm/s"),
    "density_kg_m3": ("density", "Density", "kg/m³"),
    "heat_capacity_J_kgK": ("Cp", "Specific heat capacity", "J/(kg K)"),
    "conductivity_W_mK": ("k", "Thermal conductivity", "W/(m K)"),
    "melting_temperature_K": ("melting T", "Melting temperature", "K"),
    "beam_diameter_um": ("beam D", "Beam diameter", "µm"),
    "layer_thickness_um": ("layer thickness", "Layer thickness", "µm"),
    "hatch_spacing_um": ("Hatch spacing", "Hatch spacing", "µm"),
}
ELEMENTAL = [
    "elem_atomic_number",
    "elem_atomic_volume",
    "elem_ionization_energy",
    "elem_heat_of_fusion",
    "elem_electron_affinity",
]


def required(task):
    if task not in SPEC["tasks"]:
        raise ValueError("Unknown output")
    features = SPEC["tasks"][task]["features"]
    return [name for name, (feature, _, _) in FIELDS.items() if feature in features]


def prepare(task, raw):
    """Do not infer units or substitute unverified material properties."""
    expected = {"alloy", "process"} | set(required(task))
    if task == "width":
        expected.add("composition_wt_pct")
    if set(raw) != expected:
        raise ValueError(
            f"Input fields differ: missing {sorted(expected-set(raw))}; unexpected {sorted(set(raw)-expected)}"
        )
    if raw["process"] != "SLM":
        raise ValueError("Only the frozen SLM process scope is supported")
    if not isinstance(raw["alloy"], str) or not raw["alloy"].strip():
        raise ValueError("An alloy identity is required")
    out = {}
    warnings = []
    for name in required(task):
        v = raw[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a finite positive number in the stated unit")
        out[FIELDS[name][0]] = float(v)
    if out["melting T"] <= 298:
        raise ValueError("Melting temperature must exceed the inherited 298 K ambient reference")
    if task in {"width", "length"}:
        out.update(proc_PBF=1.0, proc_DED=0.0)
    if task == "width":
        composition = raw["composition_wt_pct"]
        if (
            not isinstance(composition, dict)
            or not composition
            or set(composition) - set(CONSTANTS["ELEMENT_PROPS"])
        ):
            raise ValueError(
                "Composition must name supported elements and their weight percentages"
            )
        for el, value in composition.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 100
            ):
                raise ValueError(f"Invalid weight percentage for {el}")
        total = sum(composition.values())
        if total > 101:
            raise ValueError("Composition weight percentages exceed 101%")
        if not 99 <= total <= 101:
            warnings.append(
                f"Composition totals {total:.2f}%; unspecified fractions are zero under the inherited representation."
            )
        weighted = sum(
            (
                float(composition.get(el, 0)) / 100 * np.array(values)
                for el, values in CONSTANTS["ELEMENT_PROPS"].items()
            ),
            np.zeros(5),
        )
        out.update(dict(zip(ELEMENTAL, weighted.tolist())))
    if task == "classification":
        if raw["alloy"] not in ETA:
            raise ValueError("No audited minimum-absorptivity coefficient exists for this alloy")
        eta = ETA[raw["alloy"]]
        x = (
            0.6
            * eta
            * out["Power"]
            / (
                (out["melting T"] - 298)
                * np.pi
                * out["density"]
                * out["Cp"]
                * (out["Velocity"] / 1000)
                * (out["beam D"] / 2e6) ** 2
            )
        )
        out["absorptivity_1"] = float(-0.7 * np.expm1(-x))
    if set(out) != set(SPEC["tasks"][task]["features"]) or not all(
        math.isfinite(v) for v in out.values()
    ):
        raise ValueError("Prepared features are incomplete or nonfinite")
    return out, warnings


def raw_from_row(task, row_id):
    row = (CLS if task == "classification" else REG).iloc[int(row_id)]
    raw = {"alloy": str(row.Material).strip(), "process": "SLM"}
    for name in required(task):
        raw[name] = float(row[FIELDS[name][0]])
    if task == "width":
        raw["composition_wt_pct"] = {}
        for el in CONSTANTS["ELEMENT_PROPS"]:
            col = next(x for x in [f"{el} (wt%)", f"{el} (wt.%)"] if x in row.index)
            value = pd.to_numeric(row[col], errors="coerce")
            raw["composition_wt_pct"][el] = float(value) if pd.notna(value) else 0.0
    return raw
