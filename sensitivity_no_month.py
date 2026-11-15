"""Sensitivity check: drop Month, retune LR and XGBoost by the same CV, with and without PageValues.

Training set only (same seed-42 split and folds as the main analysis). The test set is never scored.
"""
import sys
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 42
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

df = pd.read_csv(sys.argv[1])
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)
X, y = df.drop(columns="Revenue"), df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
assert len(X_train) == 9864


def lr_model():
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=5000)
    return LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)


def pipeline(kind, columns):
    cat = [c for c in CATEGORICAL if c in columns]
    num = [c for c in columns if c not in cat]
    if kind == "lr":
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)])
        return Pipeline([("prep", prep), ("model", lr_model())]), {"model__C": np.logspace(-3, 2, 11)}
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)])
    model = XGBClassifier(subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4)
    grid = {"model__n_estimators": [100, 300, 600], "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.03, 0.1]}
    return Pipeline([("prep", prep), ("model", model)]), grid


out = {}
for kind in ["lr", "xgb"]:
    for version in ["with", "without"]:
        drop = ["Month"] + (["PageValues"] if version == "without" else [])
        Xtr = X_train.drop(columns=drop)
        pipe, grid = pipeline(kind, list(Xtr.columns))
        s = GridSearchCV(pipe, grid, scoring="average_precision", cv=CV).fit(Xtr, y_train)
        i = s.best_index_
        folds = [round(float(s.cv_results_[f"split{k}_test_score"][i]), 4) for k in range(5)]
        out[f"{kind}_{version}"] = {
            "best_params": {k.split("__")[1]: (float(v) if not isinstance(v, int) else v) for k, v in s.best_params_.items()},
            "fold_ap": folds,
        }
        f = np.array(folds)
        print(kind, version, out[f"{kind}_{version}"]["best_params"],
              f"mean {f.mean():.4f} SD {f.std(ddof=1):.4f} SE {f.std(ddof=1)/np.sqrt(5):.4f}")

json.dump(out, open(sys.argv[2], "w"), indent=1)
