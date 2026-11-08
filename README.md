# Online Shoppers Purchase Intention

ESE 4020 course project: how much does the `PageValues` feature add to predicting whether an online shopping session ends in a purchase?

## Data
[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (Sakar & Kastro, 2018), licensed under CC BY 4.0. The data file is not included in this repository.

## How to run
1. Download the zip from the UCI page and unzip it.
2. Put `online_shoppers_intention.csv` in the same folder as the notebook.
3. Run `baseline_checkpoint_results.ipynb` from top to bottom. If the CSV is missing, the notebook downloads the data with `ucimlrepo` (`pip install ucimlrepo`).

Requirements: Python 3 with numpy, pandas, scikit-learn and matplotlib.

## Files
- `baseline_checkpoint_results.ipynb`: baseline checkpoint (lasso logistic regression with and without `PageValues`, stratified 5-fold CV, scored by average precision)
- `baseline.png`: CV average precision with and without `PageValues`

The 20% test set is held out and is not used until the final evaluation.
