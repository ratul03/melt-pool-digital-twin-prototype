"""Rebuild the main results of the thesis from this package and check them against the stored copies.

Usage
    python reproduce.py                  # depth, width and length (about 10 minutes)
    python reproduce.py --task depth     # one output only (quicker)
    python reproduce.py --work <folder>  # folder for working files (default: ./reproduce_work)

What it does
  1. Copies the frozen inputs of the main design (Chapter 5) into a working folder, using the
     folder layout that the study code expects. The study code itself is not changed: it checks
     the SHA-256 hash of every frozen file before it runs.
  2. Reruns the main experiment (code/main_design/03_run_experiment.py) from the cached frozen-model
     predictions and the fixed partitions, then summarises it (04_summarise.py).
  3. Reruns the keyhole check of Section 7.1.3 (code/keyhole_check.py).
  4. Compares every rebuilt file with the copy stored under results/ and prints PASS or FAIL.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
STORED = PKG / "results/chapter6_main_design/main_design_runs"


def _recorded_hashes():
    """SHA-256 of every package file, from CHECKSUMS.sha256."""
    out = {}
    path = PKG / "CHECKSUMS.sha256"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            out[name] = digest
    return out


RECORDED = _recorded_hashes()


def copy(src, dst):
    """Copy a file and restore the exact bytes recorded in CHECKSUMS.sha256.

    Git can change line endings (LF or CRLF) when the repository is uploaded or downloaded.
    The study code checks the SHA-256 hash of its inputs, so the original line endings are restored here.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = Path(src).read_bytes()
    want = RECORDED.get(Path(src).resolve().relative_to(PKG).as_posix())
    if want:
        lf = data.replace(b"\r\n", b"\n")
        for candidate in (data, lf, lf.replace(b"\n", b"\r\n")):
            if hashlib.sha256(candidate).hexdigest() == want:
                data = candidate
                break
    dst.write_bytes(data)


def build_workspace(work):
    study = work / "main_design"
    if study.exists():
        shutil.rmtree(study)
    for f in (PKG / "code/main_design").glob("*.py"):
        copy(f, study / "code" / f.name)
    copy(PKG / "code/main_design/inputs/prepare.py", study / "inputs/prepare.py")
    p = PKG / "data/protocols"
    copy(p / "main_design_protocol.json", study / "protocol/confirmatory_spec.json")
    copy(p / "main_design_bound_correction.json", study / "protocol/correction_2026-11-02.json")
    copy(p / "main_design_partitions.json", study / "protocol/partitions.json")
    copy(p / "main_design_partition_cohorts.csv", study / "protocol/partition_cohorts.csv")
    for f in ("ml_predictions.csv", "source_calibration_scores.csv", "model_registry.csv"):
        copy(PKG / "data/main_design" / f, study / "data" / f)
    # Files inherited from the benchmark study; the study code looks for them under this folder name.
    inh = work / "Thesis_C_Results_2026-10-20/Thesis_C_Weeks_1_2"
    copy(p / "benchmark_protocol.json", inh / "protocol/frozen.json")
    copy(p / "benchmark_splits.json", inh / "protocol/splits.json")
    for t in ("depth", "width", "length"):
        copy(PKG / "data/prepared" / f"{t}.csv", inh / "data" / f"{t}.csv")
    return study


def same_table(a, b):
    x = pd.read_csv(a, float_precision="round_trip")
    y = pd.read_csv(b, float_precision="round_trip")
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False, "different shape"
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(
                x[c].to_numpy(float), y[c].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True
            ):
                return False, "column " + c
        elif not (x[c].astype(str) == y[c].astype(str)).all():
            return False, "column " + c
    return True, ""


def same_json(a, b):
    def close(u, v):
        if isinstance(u, dict):
            return (
                isinstance(v, dict) and u.keys() == v.keys() and all(close(u[k], v[k]) for k in u)
            )
        if isinstance(u, list):
            return (
                isinstance(v, list) and len(u) == len(v) and all(close(i, j) for i, j in zip(u, v))
            )
        if isinstance(u, float) or isinstance(v, float):
            return abs(float(u) - float(v)) <= 1e-9
        return u == v

    return close(json.loads(Path(a).read_text()), json.loads(Path(b).read_text())), ""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=["depth", "width", "length", "all"], default="all")
    ap.add_argument("--work", default=str(PKG / "reproduce_work"))
    a = ap.parse_args()
    work = Path(a.work).resolve()
    tasks = ["depth", "width", "length"] if a.task == "all" else [a.task]
    study = build_workspace(work)
    print("Working folder:", work)

    for t in tasks:
        print(f"\n[1/3] Main experiment, {t}")
        subprocess.run([sys.executable, "code/03_run_experiment.py", t], cwd=study, check=True)
    checks = []
    for f in sorted((study / "results/conditions").glob("*.csv")):
        checks.append(("conditions/" + f.name,) + same_table(f, STORED / "conditions" / f.name))

    if a.task == "all":
        print("\n[2/3] Summary tables")
        subprocess.run([sys.executable, "code/04_summarise.py"], cwd=study, check=True)
        for f in sorted((study / "results").glob("*")):
            if f.is_file() and (STORED / f.name).exists():
                ok = (
                    same_json(f, STORED / f.name)
                    if f.suffix == ".json"
                    else same_table(f, STORED / f.name)
                )
                checks.append((f.name,) + ok)
    else:
        print("\n[2/3] Summary tables skipped (they need all three outputs)")

    if "depth" in tasks:
        print("\n[3/3] Keyhole check (Section 7.1.3)")
        kdir = work / "keyhole_check"
        kdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(PKG / "code/keyhole_check.py"),
                str(study),
                str(PKG / "data/prepared/depth.csv"),
            ],
            cwd=kdir,
            check=True,
        )
        checks.append(
            ("keyhole_check_records.csv",)
            + same_table(
                kdir / "keyhole_check_records.csv",
                PKG / "results/keyhole_check/keyhole_check_records.csv",
            )
        )

    print("\nComparison with the stored results")
    for name, ok, why in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name} {why}")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} files identical to the stored results.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
