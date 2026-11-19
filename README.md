# Melt-pool prediction on new alloys with a digital twin prototype

Code, data and results for the UNSW Thesis C report *A Measurement-Updated Digital Twin Prototype for Melt-Pool
Prediction on Unseen Alloys* (Ratul Sharma, MMAN4953, November 2026).

The thesis tests published MeltpoolNet machine-learning models on alloys left out of training, and proposes a method
that corrects a frozen model with a few measurements of the new alloy (a shift or linear update) and builds a 90%
jackknife+ range around it. A digital twin prototype runs the method in a web browser.

## Quick start

Python 3.10 or later (the results were produced with 3.10.12; `reproduce.py` was also checked with 3.13).

```
python -m pip install -r requirements.txt
python reproduce.py          # rebuilds the Chapter 6 results and the keyhole check, then compares them with results/
python run_prototype.py      # starts the digital twin prototype at http://127.0.0.1:8766
```

`reproduce.py` reruns the main experiment (19 alloy cases, 950 test draws, 390,000 condition rows) from the cached
frozen-model predictions and the fixed partitions, then prints PASS or FAIL for every rebuilt file (26 files, all identical
when checked on 19 November 2026). It takes about 10 minutes; `python reproduce.py --task depth` runs depth only. Working files go to `reproduce_work/`.

## Contents

| Folder | Contents | Thesis |
|---|---|---|
| `data/raw` | MeltpoolNet regression and classification files (Akbari et al., 2022) | Ch. 3, 5 |
| `data/prepared` | Task datasets (SLM only, exclusions applied), alloy and class counts | Ch. 5, App. B |
| `data/protocols` | Benchmark protocol and splits, earlier design, main design protocol, partitions and bound correction | Ch. 3, 5 |
| `data/main_design` | Cached frozen-model predictions, source calibration scores, model registry | Ch. 6, 7 |
| `data/checks` | Data checks: fitting-set sizes, material properties, source publications | Ch. 3, 5, App. B |
| `models` | 50 frozen models (seed 0): one per held-out alloy for each output, plus one random-split model per output | Ch. 3, 4 |
| `code/benchmark` | Model fitting and held-out-alloy benchmark | Ch. 3 |
| `code/base_paper_reproduction` | Reproduction of the published MeltpoolNet results | Table 3.2 |
| `code/main_design` | Main experiment: partitions, prediction cache, experiment, summary, verification, figures | Ch. 5, 6 |
| `code/earlier_design`, `code/calibration_study` | Earlier fixed-test design and calibration study | App. B |
| `code/keyhole_check.py` | Keyhole check | Section 7.1.3, Table 7.2 |
| `code/report_tables` | Scripts that built the report tables from the original study folders (kept as a record) | All |
| `digital_twin_prototype` | Prototype code (`prototype/`) and its check outputs (`verification/`) | Sections 4.2, 6.1, App. A |
| `results/chapter3_benchmark` | Tables 3.1 to 3.3 and Figure 3.1 with source files | Ch. 3 |
| `results/chapter6_main_design` | Chapter 6 tables and figures; `main_design_runs/` holds every condition row | Ch. 6, 7 |
| `results/classification_unit_check` | Rerun with the melting-temperature unit error corrected | Section 6.4.5, Table B.2 |
| `results/earlier_design` | Earlier fixed-test design | App. B |
| `results/keyhole_check` | One row per depth test record at nine measurements | Table 7.2 |

Some file names in `results/` use the table numbers of an earlier draft of the thesis, so match them to the thesis
tables by their content.

## Digital twin prototype

`python run_prototype.py` starts a local web server and opens the prototype in a browser. It runs on one computer and
has no connection to a machine. Choose an output and a held-out alloy, enter or replay measurements, and the prototype
shows the frozen prediction, the updated prediction and the 90% range. From nine measurements it uses the same update
and jackknife+ code as Chapter 6 (`method_common.py` is a byte-identical copy). Sessions can be saved and reloaded.
The check outputs in `digital_twin_prototype/verification` record 11,075 replay checks and 242 identity checks, all
passed (`check_summary.json`).

## Notes and limits

- The update forms and the target-only fit were chosen after an exploratory test on the earlier design's test records.
  The main design then used new group-aware draws and a fixed protocol (`data/protocols/main_design_protocol.json`).
- The frozen main-design code clipped only the lower range limit at zero. The correction of 2 November 2026 bounds both
  limits; it changed 1,638 of 195,000 clipped condition rows, no coverage value, no update choice and not the primary
  test result (`results/chapter6_main_design/bound_correction_note.md`).
- Seeds and draws are not independent tests; the alloy is the unit of analysis.
- The study code checks the SHA-256 hash of its frozen inputs. `reproduce.py` and `run_prototype.py` therefore copy
  the inputs into the folder layout of the original studies in a working folder, without changing the code.
- `CHECKSUMS.sha256` records every file with its original line endings. Git can convert line endings when files
  are uploaded or downloaded, so a checksum check may flag text files on some systems. `reproduce.py` and
  `run_prototype.py` restore the recorded bytes before running, so the hash checks of the study code still pass.

## Data source

The raw data are the public MeltpoolNet files of Akbari et al. (2022), *MeltpoolNet: Melt pool characteristic
prediction in metal additive manufacturing using machine learning*, Additive Manufacturing 55, 102817,
https://doi.org/10.1016/j.addma.2022.102817, downloaded from https://github.com/BaratiLab/MeltpoolNet.
That repository states no licence; the files are included here unchanged for reproducibility, with credit to the
authors. Please cite the original paper when using the data.

## Licence

The code is released under the MIT licence (`LICENSE`). The data remain the work of their original authors.
