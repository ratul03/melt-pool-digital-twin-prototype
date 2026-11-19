# Corrected classification check (melting-temperature units)

Separate sensitivity analysis, made on 2 November 2026 after the frozen studies. The frozen results are not changed.

## The error

In the public MeltpoolNet classification file, 155 of the 669 prepared records give the melting temperature 273 lower than supplementary Table A.10 (kelvin):

| Alloy | Values in the data (K) | Table A.10 (K) | Records with the low value | Source papers |
|---|---|---|---|---|
| IN718 | 1400 and 1673 | 1673 | 70 | 32, 51 |
| SS316L | 1415 and 1688 | 1688 | 72 | 69 |
| Ti-6Al-4V | 1649 and 1922 | 1922 | 13 | 34, 47 |

The low values are the Celsius values. Absorptivity 1 uses (melting temperature − 298 K), so it also changes for 154 records. The last record changes by less than 0.00001, because its value is at the 0.7 upper limit of the equation. The regression files have one melting temperature for each alloy, equal to Table A.10.

## Method

Same partitions, features, model settings and seeds as the frozen studies. Step A refits on the original data and must reproduce every stored prediction: 0 label mismatches, 0 probability difference, and the 20 Oct summaries reproduce exactly. Step B refits on the corrected data.

## Results

| Measure | Frozen (published data) | Corrected |
|---|---|---|
| Random split accuracy | 0.808 | 0.819 |
| Unseen-alloy accuracy (baseline 0.473) | 0.447 | 0.424 |
| Unseen-alloy macro-F1 (baseline 0.157) | 0.256 | 0.252 |
| Class-set coverage, calibration on other alloys | 0.450 | 0.357 |
| Class-set coverage, 9 measurements | 0.883 | 0.884 |
| Mean set size, 9 measurements (of 4) | 3.35 | 3.22 |
| Class-set coverage, 20 measurements | 0.896 | 0.902 |

No conclusion changes. The classifier stays below the most-frequent-class baseline on unseen alloys, and the class sets stay large.
