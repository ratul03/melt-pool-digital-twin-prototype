"""Freeze supplementary comparisons before calculating their outcomes."""

from datetime import datetime, timezone
from common import ROOT, BASE, W12, W34, sha, write_json

path = ROOT / "protocol/supplementary_spec.json"
if path.exists():
    raise SystemExit("Existing specification retained; use the checked experiment runner.")
dependencies = {}
for folder in [
    W12 / "data",
    W12 / "protocol",
    W12 / "models",
    W34 / "results",
    W34 / "protocol",
    W34 / "models",
]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix in {".csv", ".json", ".joblib"}:
            dependencies[p.relative_to(BASE).as_posix()] = sha(p)
for p in [
    W34 / "verification/feature_overlap_pairs.csv",
    W12 / "inputs/code/reference_constants.json",
    W12 / "inputs/code/reproduce.py",
]:
    dependencies[p.relative_to(BASE).as_posix()] = sha(p)
write_json(
    path,
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "supplementary_exploratory_fixed_before_new_comparator_results",
        "reason": "Complete the agreed results scope; not an original preregistered hypothesis.",
        "primary_task": "depth",
        "secondary_tasks": ["width", "length", "classification"],
        "predictors": {
            "ml": "Unchanged saved predictions from the frozen Weeks 3 and 4 experiment.",
            "simple": "Training-only arithmetic mean for geometry; training-only class frequencies and lexicographic majority class for classification.",
        },
        "calibration": "Identical absolute residual / 1-p_true scores and finite-sample 90% rank for both predictors; source-only at budget0, target-only at9 and20.",
        "partitions": "Exact frozen train, source-calibration, target selection and final-test records; five seeds; twenty nested selections.",
        "cohorts": "primary_paired and secondary_9_only retained separately; all eligible alloys included.",
        "comparisons": [
            "ML minus simple at each budget",
            "target9 and target20 minus source within predictor",
        ],
        "geometry_metrics": [
            "MAE",
            "RMSE",
            "R2",
            "coverage",
            "width",
            "interval_score",
            "lower_miss_penalty",
            "upper_miss_penalty",
            "negative_prediction_fraction",
            "negative_lower_fraction",
        ],
        "classification_metrics": [
            "accuracy",
            "macro_f1_fixed_four_labels",
            "coverage",
            "set_size",
            "empty_fraction",
            "singleton_fraction",
            "full_fraction",
        ],
        "aggregation": "Test records within each condition, then repeats within seed, seeds within alloy, equal alloy weights. Repeats and seeds are descriptive variability, not independent trials.",
        "sensitivity": "Same previously defined active-feature overlap exclusions, applied to both predictors and all budgets.",
        "hypothetical_usefulness": {
            "half_width_tolerances_um": [25, 50, 100, 150, 200, 300, 500, 750, 1000],
            "meaning": "State fraction with q<=tolerance, coverage among those states, and covered-and-narrow fraction. Hypothetical precision thresholds, not manufacturing acceptance limits.",
            "no_selection": "No tuning or selective row-level abstention; the regression margin is constant within a calibration state.",
        },
        "diagnostics": [
            "interval_score_decomposition",
            "per_alloy_paired_differences",
            "seed_and_selection_spread",
            "signed_residuals",
            "source_alloy_overlap",
            "negative_bounds",
            "classification_set_distribution",
            "baseline_transfer",
        ],
        "no_changes": [
            "point model training",
            "hyperparameters",
            "physical clipping",
            "frozen test partitions",
        ],
        "sources": [
            {
                "citation": "Angelopoulos and Bates (2022)",
                "url": "https://arxiv.org/abs/2107.07511",
            },
            {
                "citation": "Gneiting and Raftery (2007)",
                "url": "https://doi.org/10.1198/016214506000001437",
            },
        ],
        "dependencies": dependencies,
        "runner_sha256_at_freeze": sha(ROOT / "code/run_comparisons.py"),
    },
)
print(f"Frozen supplementary specification; {len(dependencies)} dependencies.")
