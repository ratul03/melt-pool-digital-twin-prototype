# Correction of 2 November 2026: interval clipping

This folder is a corrected copy of `Thesis results C 22-10-2026`. The original folder is not changed.

## The defect

The frozen runner clipped only the lower bound of each interval at 0. When the upper bound was also below 0, the clipped interval was inverted (lower bound above upper bound), and its width was negative. An independent audit found this on 2 November 2026.

- 1,638 of 195,000 clipped condition rows changed. 10 rows had a negative mean width.
- Coverage did not change in any row. The automatic update choices did not change.
- Unclipped results are identical to the frozen results.

## The correction

Clipped intervals now project both bounds onto [0, ∞): lower = max(lower, 0) and upper = max(upper, 0). If both bounds are below 0, the interval becomes [0, 0]. `metrics()` now refuses any inverted interval, and `05_verify.py` checks interval order and nonnegative bounds.

This is a correction made after the results were seen. The partitions, cached predictions, methods, hypotheses, primary test and decision rules did not change. `protocol/correction_2026-11-02.json` records the old and new hashes of the two changed frozen files.

## Effect on the main results

| Depth, 9 measurements | Frozen interval score (µm) | Corrected (µm) |
|---|---|---|
| Calibration on other alloys | 1153.35 | 1142.59 |
| ML + target calibration (20 Oct method) | 625.26 | 624.14 |
| Target-only fit on power and speed | 464.36 | 451.60 |
| ML + shift update | 568.06 | 566.80 |
| ML + automatic update (proposed) | 440.67 | 438.79 |

The primary test does not change: the proposed method is better on 3 of 7 alloys, exact two-sided Wilcoxon p = 0.8125. H3 is still not supported. `results/frozen_vs_corrected_summary.csv` lists every change.

## Folders

- `results`, `figures`, `verification`: corrected outputs (29,402 checks passed, 0 failed).
- `frozen_outputs_2026-10-22`: the original frozen outputs, unchanged.
