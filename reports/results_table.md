| Model | Evaluated on | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Predict "rejected" for everyone (reference) | 5-fold CV on training data, mean ± std | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.778 ± 0.000 |
| Rule: reject if previous default (reference) | 5-fold CV on training data, mean ± std | 0.826 ± 0.003 | 0.622 ± 0.004 | 0.451 ± 0.004 | 1.000 ± 0.000 | 0.730 ± 0.005 |
| Random Forest, all features, threshold 0.50 (baseline) | 5-fold CV on training data, mean ± std | 0.973 ± 0.002 | 0.823 ± 0.010 | 0.891 ± 0.012 | 0.766 ± 0.013 | 0.927 ± 0.004 |
| HistGradientBoosting, no gender, threshold 0.50 | 5-fold CV on training data, mean ± std | 0.977 ± 0.002 | 0.834 ± 0.010 | 0.888 ± 0.007 | 0.787 ± 0.020 | 0.930 ± 0.003 |
| **HistGradientBoosting, no gender, threshold 0.78 (final)** | **held-out test, 95% bootstrap CI** | **0.977** [0.974, 0.979] | **0.782** [0.766, 0.797] | **0.968** [0.959, 0.977] | **0.655** [0.634, 0.676] | **0.919** [0.913, 0.924] |

Test rows: 8,999. Training rows: 35,994.

Audit by gender (held-out test):

| Gender | Rows | Approved in data | Approved by model | Recall | False approval rate |
|---|---|---|---|---|---|
| female | 4,005 | 21.8% | 14.8% | 65.2% | 0.7% |
| male | 4,994 | 22.6% | 15.3% | 65.8% | 0.5% |

Largest gap in model approval rate: 0.5%. Largest gap in recall: 0.7%.

Audit by age band (held-out test):

| Age band | Rows | Approved in data | Approved by model | Recall | False approval rate |
|---|---|---|---|---|---|
| 24 or under | 3,241 | 24.1% | 17.3% | 70.3% | 0.5% |
| 25-29 | 3,207 | 21.8% | 14.6% | 64.4% | 0.6% |
| 30-39 | 2,131 | 19.8% | 12.4% | 60.0% | 0.6% |
| 40 and over | 420 | 22.9% | 14.5% | 59.4% | 1.2% |

Largest gap in model approval rate: 4.9%. Largest gap in recall: 10.9%.
