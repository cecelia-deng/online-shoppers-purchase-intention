# Online Shoppers Purchase Intention

ESE 4020 course project: how much does the `PageValues` feature add to
predicting whether an online shopping session ends in a purchase?

## Data

We use the
[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online)
(Sakar & Kastro, 2018), licensed under CC BY 4.0. The raw CSV is not included
in this repository. Download `online_shoppers_intention.csv` from UCI and put
it in the repository root before running the analysis.

## Project stages

### Baseline checkpoint

The original checkpoint files remain unchanged:

- `baseline_checkpoint_results.ipynb`: lasso logistic regression with and
  without `PageValues`, stratified five-fold cross-validation, and average
  precision scoring.
- `baseline.png`: checkpoint plot of cross-validation average precision.

The checkpoint used only the training portion of a stratified 80/20 split;
the test portion was held out at that stage.

### Final logistic-regression analysis

`pagevalues_logistic_analysis.py` extends the checkpoint with a data audit,
exploratory figures, leakage-safe cross-validation, out-of-fold threshold
selection, a one-time held-out evaluation, coefficient tables, and a paired
bootstrap comparison. Generated files are stored under `results/`.

## Final held-out results

- With `PageValues`: AP 0.623129, accuracy 0.875912, F1 0.632212.
- Without `PageValues`: AP 0.326114, accuracy 0.726277, F1 0.408414.
- AP difference (with minus without): 0.297014.
- Paired-bootstrap 95% interval for the AP difference: [0.238985, 0.354765].

The logistic-regression settings and F1 thresholds were selected using only
the training data before the test set was evaluated. The paired bootstrap
used 2,000 repetitions and random seed 42.

## Repository contents

```text
.
├── README.md
├── baseline_checkpoint_results.ipynb
├── baseline.png
├── pagevalues_logistic_analysis.py
├── requirements.txt
└── results/
    ├── run_manifest.json
    ├── figures/
    └── tables/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Reproduce the baseline checkpoint

Open `baseline_checkpoint_results.ipynb` and run all cells from top to bottom.
If the CSV is missing, the notebook can retrieve the dataset through
`ucimlrepo`.

## Reproduce the training-only final analysis

This command creates the audit, EDA figures, five-fold cross-validation
results, coefficient tables, and out-of-fold secondary metrics without
scoring the held-out test set:

```bash
python pagevalues_logistic_analysis.py \
  --data online_shoppers_intention.csv \
  --output-dir results
```

## Reproduce the finalized held-out analysis

This command reproduces the recorded held-out evaluation and paired
bootstrap after all logistic-regression choices have been frozen:

```bash
python pagevalues_logistic_analysis.py \
  --data online_shoppers_intention.csv \
  --output-dir results \
  --run-final-test \
  --bootstrap-repetitions 2000
```

## Reproducibility settings

- Stratified 80/20 train/test split with random seed 42.
- Shuffled stratified five-fold cross-validation with seed 42.
- Eleven `C` values from 0.001 to 100 on a logarithmic grid.
- Average precision as the primary selection metric.
- One-hot encoding and numerical standardization fitted inside each fold.
- F1 thresholds selected from training-only out-of-fold predictions.
