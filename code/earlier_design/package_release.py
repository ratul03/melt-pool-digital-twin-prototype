"""One dated, self-contained research-output archive, with no thesis prose."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import platform
import zipfile
import importlib.metadata
import pandas as pd
from common import ROOT, BASE, W12, W34, sha, write_json, read_csv

DATE = "2026-10-20"
RELEASE = "Thesis_C_Results_" + DATE


def main():
    for name in ["comparison_checks", "prototype_checks", "diagnostic_checks"]:
        report = json.loads((ROOT / f"verification/{name}.json").read_text())
        assert report["failed"] == 0, name
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "matplotlib",
                "threadpoolctl",
            ]
        },
    }
    write_json(ROOT / "verification/release_environment.json", env)
    summary = read_csv(ROOT / "results/summary.csv")
    primary = summary[summary.cohort == "primary_paired"]
    paired = read_csv(ROOT / "results/paired_differences.csv")
    headlines = []
    for task in ["depth", "width", "length"]:
        z = primary[(primary.task == task) & (primary.budget == 20)].set_index("predictor")
        delta = z.loc["ml", "mean_interval_score"] - z.loc["simple", "mean_interval_score"]
        by = paired[
            (paired.task == task) & (paired.cohort == "primary_paired") & (paired.budget == 20)
        ]
        headlines.append(
            {
                "task": task,
                "budget": 20,
                "ml_interval_score_um": z.loc["ml", "mean_interval_score"],
                "simple_interval_score_um": z.loc["simple", "mean_interval_score"],
                "ml_minus_simple_interval_score_um": delta,
                "ml_coverage": z.loc["ml", "coverage"],
                "simple_coverage": z.loc["simple", "coverage"],
                "n_alloys": int(z.loc["ml", "n_alloys"]),
                "alloys_favouring_ml_interval_score": int(
                    (by.mean_interval_score_ml_minus_simple < 0).sum()
                ),
            }
        )
    write_json(ROOT / "results/key_comparisons.json", headlines)
    index = []
    for p in sorted((ROOT / "results").glob("*.csv")):
        frame = read_csv(p)
        index.append(
            {
                "file": p.relative_to(ROOT).as_posix(),
                "rows": len(frame),
                "columns": list(frame.columns),
                "sha256": sha(p),
            }
        )
    write_json(ROOT / "results/index.json", index)
    write_json(
        ROOT / "protocol/release_scope.json",
        {
            "release_date": DATE,
            "requested_scope": "Code, experiments, figures and results only",
            "completed": [
                "calibrated_simple_comparators_all_four_outputs",
                "paired_per_alloy_analysis",
                "uncertainty_score_decomposition",
                "input_overlap_sensitivity_both_predictors",
                "hypothetical_precision_analysis",
                "selection_variability",
                "signed_residual_diagnostics",
                "original_point_transfer_tables",
                "raw_input_adapter_all_four_outputs",
                "interactive_measurement_and_state_workflow",
                "numerical_and_prototype_verification",
            ],
            "experimental_status": "Supplementary exploratory comparisons using previously evaluated frozen test records.",
            "external_validation": False,
            "live_machine_connection": False,
            "validated_manufacturing_tolerances": False,
            "further_model_selection_on_test_labels": False,
            "physical_geometry_clipping": False,
            "training": "Original model fits retained; no architecture or hyperparameter search added.",
            "statistical_units": "Alloys are the limited transfer units; seeds and selection repeats are dependent.",
            "coverage_storage": "Inherited evaluated inclusion booleans preserved at numerical endpoint ties; independent checks quantify all discrepancies.",
            "scope_boundary": "Completes the agreed computational experiments and prototype. Does not establish manufacturing readiness or promise a thesis grade.",
        },
    )
    # A local results browser, not a report: numbers, plots and direct data links.
    sections = []
    for task in ["depth", "width", "length", "classification"]:
        cols = ["budget", "predictor", "n_alloys", "coverage"] + (
            ["accuracy", "macro_f1", "mean_set_size", "full_set_fraction"]
            if task == "classification"
            else ["MAE", "RMSE", "mean_interval_score", "mean_width"]
        )
        sections.append(
            f"<h2>{task.title()}</h2>"
            + primary[primary.task == task][cols].to_html(
                index=False, float_format=lambda v: f"{v:.4f}", border=0
            )
        )
    figs = json.loads((ROOT / "figures/index.json").read_text())
    pictures = "".join(
        f'<details><summary>{html.escape(f["description"])}</summary><img src="{f["png"]}" alt="{html.escape(f["description"])}"><p><a href="{f["svg"]}">SVG</a> · <a href="{f["pdf"]}">PDF</a></p></details>'
        for f in figs
    )
    tables = "".join(
        f'<li><a href="{r["file"]}">{Path(r["file"]).name}</a> — {r["rows"]:,} rows</li>'
        for r in index
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Thesis C results · {DATE}</title>
<style>body{{font:15px Segoe UI,Arial,sans-serif;color:#1a3145;background:#f4f7fa;max-width:1250px;margin:30px auto;padding:0 24px}}
h1{{font-size:30px}}h2{{margin-top:30px}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dce3e9}}th{{background:#e4edf4}}details{{background:white;padding:16px;margin:12px 0}}summary{{cursor:pointer;font-weight:600}}img{{width:100%;height:auto}}a{{color:#1c6096}}p{{line-height:1.6}}.notice{{padding:14px;background:#e7eff6;border-left:4px solid #236794}}</style>
<h1>Thesis C computational results · {DATE}</h1>
<p class="notice">Paired cohorts shown below. Geometry units: µm. Coverage and other fractions range from 0 to 1.
Simple predictor: training mean for geometry; training class frequencies for classification. The numerical tables retain the separate 9-only cohorts.</p>
<p><a href="START_PROTOTYPE.ps1">Start interactive prototype</a> · <a href="REBUILD_RESULTS.ps1">Rebuild results</a> ·
<a href="SETUP_ENVIRONMENT.ps1">Set up Python 3.12 environment</a> · <a href="protocol/release_scope.json">Scope and provenance</a></p>
<p>Open the PowerShell scripts from the extracted folder. A Python 3.12 environment with the locked dependencies is required to run the models; tables and figures open directly.</p>
{''.join(sections)}<h2>Figures</h2>{pictures}<h2>Numerical tables</h2><ul>{tables}</ul>
<h2>Verification</h2><ul><li><a href="verification/comparison_checks.json">Comparison arithmetic and separation</a></li>
<li><a href="verification/prototype_checks.json">Input, replay and persistence checks</a></li>
<li><a href="verification/diagnostic_checks.json">Derived result checks</a></li></ul></html>"""
    (ROOT / f"OPEN_RESULTS_{DATE}.html").write_text(page, encoding="utf-8")
    # Dependency closure required by frozen hashes, raw feature preparation,
    # inherited point tables, saved models and the observation workflow.
    paths = {}

    def include(p):
        if p.is_file():
            paths[p.relative_to(BASE).as_posix()] = p

    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not any(
            x in rel for x in ["__pycache__", "matplotlib_cache", "saved_sessions", ".corrupt.json"]
        ):
            if p.name not in {"release_manifest.json", "archive_check.json"}:
                include(p)
    for package in [W12, W34]:
        for folder in ["data", "inputs", "protocol", "models", "results", "figures"]:
            for p in (package / folder).rglob("*"):
                if (
                    p.is_file()
                    and "__pycache__" not in str(p)
                    and p.suffix in {".csv", ".json", ".joblib", ".py", ".png", ".svg"}
                ):
                    include(p)
        include(package / "requirements-lock.txt")
        for p in (package / "verification").glob("*.json"):
            include(p)
    for filename in ["workflow.py", "test_workflow.py", "run_demo.py"]:
        include(W34 / "prototype" / filename)
    for filename in [
        "feature_overlap_pairs.csv",
        "feature_overlap_counts.csv",
        "sensitivity_summary.csv",
        "sensitivity_per_alloy.csv",
    ]:
        include(W34 / "verification" / filename)
    for filename in ["run_baselines.py", "verify_and_summarize.py", "equal_alloy_summary.py"]:
        include(W12 / "code" / filename)
    # Rebuild entrypoint uses this release's independent checks. Old whole-folder
    # verifiers also hash excluded Word/report assets and are not portable here.
    # Historical reproduction results are retained as a separate evidence set.
    audited = BASE / "Thesis_B_Audited"
    for folder in ["results", "audit"]:
        for p in (audited / folder).rglob("*"):
            if p.is_file() and p.suffix in {".csv", ".json"}:
                include(p)
    dependencies = json.loads((ROOT / "protocol/supplementary_spec.json").read_text())[
        "dependencies"
    ]
    assert set(dependencies) <= set(paths), sorted(set(dependencies) - set(paths))
    # Also satisfy all inherited feature-data hashes used by create_replay().
    frozen = json.loads((W12 / "protocol/frozen.json").read_text())
    for rel in frozen["input_hashes"]:
        assert (W12 / rel).relative_to(BASE).as_posix() in paths, rel
    manifest = {
        "date": DATE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            rel: {"sha256": sha(p), "bytes": p.stat().st_size} for rel, p in sorted(paths.items())
        },
        "open_first": f"Thesis_C_Final_Results/OPEN_RESULTS_{DATE}.html",
        "excluded": [
            "thesis report prose",
            "Word reports",
            "virtual environments",
            "temporary caches",
            "deliberately corrupted test-state fixtures",
        ],
    }
    write_json(ROOT / "release_manifest.json", manifest)
    archive = BASE / (RELEASE + ".zip")
    # The dated root is the single folder a user sees after extraction.
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for rel, p in sorted(paths.items()):
            z.write(p, f"{RELEASE}/{rel}")
        z.write(ROOT / "release_manifest.json", f"{RELEASE}/release_manifest.json")
        z.writestr(
            f"{RELEASE}/OPEN_RESULTS_{DATE}.html",
            f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=Thesis_C_Final_Results/OPEN_RESULTS_{DATE}.html"><a href="Thesis_C_Final_Results/OPEN_RESULTS_{DATE}.html">Open the results and figures</a>',
        )
    verified = 0
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel, record in manifest["files"].items():
            assert hashlib.sha256(z.read(f"{RELEASE}/{rel}")).hexdigest() == record["sha256"], rel
            verified += 1
    result = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "files_hash_verified": verified,
        "sha256": sha(archive),
        "passed": True,
    }
    write_json(ROOT / "verification/archive_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
