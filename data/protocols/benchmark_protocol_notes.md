# Week 1 scope and experiment specification

Prepared 23 September 2026. This is the implemented technical protocol, not evidence of supervisor approval. Model settings, eligibility rules and all partitions were written to `protocol/frozen.json` before fitting any Week 2 models. Input and partition hashes make later changes detectable.

## Research direction

Develop a computational prototype integrating depth, width, length and four-class defect predictions. Test in later weeks whether existing measurements deliberately withheld from training can improve uncertainty estimates when revealed for calibration. Keep point predictors fixed during each calibration comparison. The four outputs use separate datasets and must not be presented as jointly validated measurements of one track.

Weeks 1 and 2 establish scope, targeted prior work, data feasibility, partitions and point-prediction baselines. Conformal intervals, category sets, their updates and the prototype interface are later deliverables. No new experiments or machinery are required. Six development weeks and four report weeks remain the planning structure; actual calendar deadlines were not supplied.

## Model selection fixed before scoring

| Output | Primary model | Inputs and reason |
|---|---|---|
| Depth | Neural network, 64/32/64 hidden layers, alpha 0.065533 | Power, speed, density, heat capacity, conductivity, melting temperature, beam diameter and layer thickness. Retains the Thesis B NN architecture with a pre-build feature set. This is not the paper's headline absorptivity-2 configuration. |
| Width | Gradient boosting, 495 estimators | Baseline numerical properties, process indicators and five historical weighted elemental properties. Retains the reproduced headline representation, with SLM-only data. |
| Length | Gradient boosting, 353 estimators | Baseline numerical properties, process indicators and hatch spacing. Retains the reproduced headline configuration, restricted to SLM. |
| Classification | Random forest, 160 trees | Baseline classification settings/properties plus equation-2 absorptivity 1, recalculated with the audited static material coefficient lookup. Four categories: LOF, balling, desirable and keyhole. |

All settings are inherited from the audited implementation, with no search against these test outcomes. Full scikit-learn parameters and versions are saved. Scalers are fitted on training rows only; the NN target scaler also uses training rows only. Model seeds are 0 to 4. The NN cap remains 20,000, as in the audited corrected branch; warnings and actual iterations are recorded.

The paper's headline depth model and best classifier remain measurement-assisted reference models already retained in Thesis B. They are not used as pre-build baselines because absorptivity 2 contains measured width. The historical classification values are not silently substituted for corrected values. Process indicators constant within SLM are retained to preserve the inherited feature layout; they supply no varying information.

## Data rules

Use the pinned public CSV files copied into `inputs/data/raw` and the audited feature builder copied into `inputs/code`. Keep raw row IDs. Restrict every primary task to `Sub-process == SLM`. Remove the same 11 suspect-unit regression rows rather than invent conversions. Drop incomplete active features/targets and record exclusions. Do not impute target measurements or join regression/classification records by row number.

Classification retains the established four-class target rather than claiming to cover spatter, which has only eight records in the full raw file. The corrected material lookup supports seven materials; other records without sufficient coefficients are excluded and listed. The lookup uses static material input information from the regression dataset, not its geometry outcomes. Thus held-out alloys have known material properties; this is not prediction for an alloy with unknown physical properties. The elemental feature representation and composition assumptions remain inherited limitations.

Source identifiers are retained exactly as strings after trimming. They are provenance labels, not verified laboratory identifiers. Classification source identifier 0 may be a placeholder and is not interpreted as a proven common laboratory. Alloy/source overlap is described using the supplied identifiers only. No unsupported alloy renaming or merging is performed.

## Week 2 partitions

**Random baseline:** approximately 64% model fitting, 16% reserved initial calibration, 20% test. The 20% test split uses seed 0 to 4; 20% of the remaining 80% is reserved for calibration using seed 1000 plus the run seed. Classification partitions are class-stratified.

**Leave one alloy out:** the entire test alloy is excluded from fitting and initial calibration. Other alloys are split 80% fitting and 20% reserved calibration, using the same seed rule. All test-alloy records are scored for the Week 2 transfer assessment. Five seeds vary training/calibration assignment and model randomness; test-alloy records remain the same. These are not five independent new-alloy experiments.

Reserved calibration labels are not used to train, tune, select models or compute Week 2 intervals. Reserving them now allows the fitted models to be used in Week 3 without changing their training partitions. This intentionally changes the training fraction from the Thesis B 80/20 reproduction. New scores are not like-for-like paper comparisons. Training-set sizes also differ between random and alloy-held-out protocols, so differences are not a pure causal estimate of alloy shift alone.

Groups with at least 20 test records are primary. Groups with 2 to 19 records are scored but labelled exploratory. Singleton groups would be excluded from R² evaluation. Report every eligible group; do not select on favorable performance. Undefined R² for constant targets is left missing. No clipping of negative predictions or test-score tuning is allowed.

## Future observation budgets fixed now

For each alloy with at least 29 records, reserve 20 fixed final evaluation records with seed 20260923. The remaining target-alloy rows form the observation pool. Freeze 20 repeat selections with seeds 71000 to 71019; the 9-observation subset is nested within the 20-observation subset when available. Primary 9-versus-20 comparisons require at least 40 records and use exactly the same alloys and final test rows at both budgets. Nine-only alloys are secondary, not mixed into that budget curve. Forty-plus-30 eligibility is counted but not selected as a core experiment.

The Week 2 whole-alloy assessment includes those records, but no choices will be tuned to their scores. Week 3 must keep the frozen final test labels out of the recalibration pool. A calibration-label budget is not a count of independent experiments; records from a single source can be correlated. Check observed classifier class representation without forcing the random reveal pool to contain every class after seeing results.

## Metrics and interpretation

Regression: MAE, RMSE, R², negative prediction counts, and a training-mean dummy baseline. Negative R² is relative to the test-mean reference in that metric, not necessarily the training-mean dummy. Classification: accuracy, macro-F1 over the fixed four labels (zero for an absent label), balanced accuracy over labels present in that test set, confusion matrices, and a training-majority dummy baseline. This explicit macro-F1 convention avoids changing the metric's class universe between folds; single-class exploratory folds require caution.

Save per-row predictions, probabilities, split IDs, scaler parameters, warnings and fitted seed-0 models. Aggregate row-weighted results and equal-alloy results separately. Headline random/grouped comparisons use the common cohort of alloys with at least 20 complete records. The exact test rows differ between protocols. Mean ± sample SD describes variation across the five seeds, not confidence in unseen-alloy performance.

## Week 3 decision rules

Use one absolute-residual conformal method for geometry and one probability-score method for classification. Nominal coverage is 90%; nine records is only the finite residual-interval minimum, not a reliable sample-size recommendation. Show coverage, interval width or set size and per-alloy outcomes. Assess improvement without assuming calibration repairs a poor point predictor. Tiny eligible alloy counts limit any broad generalization claim. No abstention, inverse search, retrained point model or new hyperparameter sweep is part of this core specification.
