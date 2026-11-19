"""Freeze the confirmatory specification BEFORE any interval or metric is computed.

Run order: 01_make_partitions.py -> 02_cache_predictions.py -> 00_freeze_spec.py -> 03 -> 04 -> 05 -> 06.
03_run_experiment.py refuses to run if any frozen file hash has changed.
"""

from datetime import datetime, timezone
from common import (
    ROOT,
    W12,
    W34,
    SRC,
    TASKS,
    ALPHA,
    BUDGETS,
    REPEATS,
    TEST_N,
    MIN_TEST_N,
    PRIMARY_MIN_ROWS,
    SECONDARY_MIN_ROWS,
    SEED_BASE,
    TARGET_ONLY_FEATURES,
    METHODS,
    sha,
    write_json,
)

path = ROOT / "protocol/confirmatory_spec.json"
if path.exists():
    raise SystemExit("Specification already frozen; refusing to replace it.")
if any((ROOT / "results").glob("conditions/*.csv")):
    raise SystemExit("Results exist; a specification cannot be frozen after results.")
frozen_files = [
    "code/common.py",
    "code/01_make_partitions.py",
    "code/02_cache_predictions.py",
    "code/03_run_experiment.py",
    "code/04_summarise.py",
    "inputs/prepare.py",
    "protocol/partitions.json",
    "data/ml_predictions.csv",
    "data/source_calibration_scores.csv",
    "data/model_registry.csv",
]
inherited = [
    "Thesis_C_Weeks_1_2/protocol/frozen.json",
    "Thesis_C_Weeks_1_2/protocol/splits.json",
    "Thesis_C_Weeks_1_2/data/depth.csv",
    "Thesis_C_Weeks_1_2/data/width.csv",
    "Thesis_C_Weeks_1_2/data/length.csv",
]
write_json(
    path,
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_any_interval_or_metric",
        "question": "With n measurements on an unseen alloy, does a frozen ML model plus a model update and jackknife+ "
        "give better 90% intervals than (a) the existing calibration methods and (b) models built from the "
        "target measurements alone?",
        "disclosure": (
            "The proposed method family (shift / linear update with jackknife+, zero clip) and the target-only "
            "linear baselines were chosen after an exploratory test on 2026-10-22 using the 20 Oct fixed test "
            "rows. Exploratory equal-alloy depth interval scores at 9 measurements were: current method + clip "
            "567, ML linear update + clip 402, target-only P+V + clip 411. This run uses 50 fresh, group-aware "
            "test draws per alloy and a fixed selection rule. Rows are the same physical records, so this "
            "reduces but cannot remove selection bias."
        ),
        "tasks": {
            "primary": "depth",
            "secondary": ["width", "length"],
            "excluded": {
                "classification": "Reported as a secondary negative result on 20 Oct; known unquarantined unit-error rows 679-681."
            },
        },
        "alpha": ALPHA,
        "budgets": BUDGETS,
        "repeats": REPEATS,
        "test_n": TEST_N,
        "min_test_n": MIN_TEST_N,
        "cohorts": {
            "primary": f">= {PRIMARY_MIN_ROWS} alloy rows",
            "secondary": f"{SECONDARY_MIN_ROWS}-{PRIMARY_MIN_ROWS - 1} rows, budget 9 only",
        },
        "partition_rule": "Condition groups = identical active model inputs (rounded 1e-9). Groups go wholly to test or pool. "
        "Greedy random fill of test to <= 20 rows. Calibration = nested prefixes of one random pool permutation.",
        "seed_base": SEED_BASE,
        "models": "Unchanged frozen excluded-alloy models, seeds 0-4 (Weeks 1-2 seed 0, Weeks 3-4 refits 1-4).",
        "methods": METHODS,
        "target_only_features": TARGET_ONLY_FEATURES,
        "selection_rule": "Auto methods choose the candidate with the lowest exact LOO MSE on calibration data; ties -> simpler. "
        "The choice runs inside every jackknife fold.",
        "physical_bound": "Primary results clip lower bounds at 0 for every method (geometry >= 0; coverage cannot fall). "
        "Unclipped results are stored as well.",
        "metrics": [
            "coverage",
            "mean_width",
            "interval_score (alpha=0.1)",
            "lower_penalty",
            "upper_penalty",
            "MAE",
            "negative_lower_fraction",
        ],
        "aggregation": "Rows -> condition; mean over repeats within (alloy, seed, budget, method); mean over seeds; "
        "equal-alloy mean within cohort. Alloy is the unit of analysis.",
        "acceptability": "A method is acceptable only if its equal-alloy coverage >= 0.85 (jackknife+ worst case is 0.80).",
        "hypotheses": {
            "H_primary": "Depth, primary cohort, budget 9, clipped: ml_auto_jk has a lower equal-alloy interval score than "
            "target_split (the 20 Oct method). Test: exact two-sided Wilcoxon signed-rank on alloy-level "
            "differences; with 7 alloys the smallest attainable p is 0.0156.",
            "H_existing": "ml_auto_jk beats source_split and target_split at every budget (all tasks). Descriptive + Wilcoxon.",
            "H_baseline": "ml_auto_jk beats mean_target_split and target_mean_jk at every budget.",
            "H_prior_value": "ml_auto_jk vs target_auto_jk: report the difference by budget; a crossover is allowed and is "
            "itself a result. No direction is assumed.",
        },
        "decision_rule_for_superiority": "Call ml_auto_jk superior to a comparator on a task and budget only if (1) its equal-alloy "
        "interval score is lower, (2) it wins in a majority of alloys, and (3) both coverages are acceptable.",
        "frozen_files": {f: sha(ROOT / f) for f in frozen_files},
        "inherited_files": {f: sha(SRC / f) for f in inherited},
        "sources": {
            "jackknife+": "Barber, Candes, Ramdas, Tibshirani (2021) Ann. Statist. 49(1), doi:10.1214/20-AOS1965",
            "split conformal": "Angelopoulos and Bates, arXiv:2107.07511",
            "interval score": "Gneiting and Raftery (2007) JASA, doi:10.1198/016214506000001437",
            "model updating": "Janssen et al. (2008) J Clin Epidemiol 61(1):76-86; Vergouwe et al. closed testing procedure",
            "grouped evaluation": "Meredig et al. (2018) Mol. Syst. Des. Eng.; Roberts et al. (2017) Ecography",
            "redundancy control": "Li, Fu, Omee, Hu (2024) npj Comput. Mater. (MD-HIT)",
            "transfer vs scratch": "Menon, Mondal, Basak (2022) SFF Symposium, SS316L -> IN718",
        },
    },
)
print("Frozen", path)
