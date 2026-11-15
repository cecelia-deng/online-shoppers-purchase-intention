"""XGBoost with and without PageValues: stratified 5-fold CV on the training set only.

Uses the same split (seed 42) and the same CV folds as the logistic regression baseline.
The test set is created but never scored here.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
GRID = {
    "model__n_estimators": [100, 300, 600],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.03, 0.1],
}

df = pd.read_csv(sys.argv[1])
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)
X, y = df.drop(columns="Revenue"), df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
assert len(X_train) == 9864 and len(X_test) == 2466


def build(columns):
    cat = [c for c in CATEGORICAL if c in columns]
    num = [c for c in columns if c not in cat]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ])
    model = XGBClassifier(
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4,
    )
    return Pipeline([("prep", prep), ("model", model)])


out = {}
for label, Xtr in [("with", X_train), ("without", X_train.drop(columns="PageValues"))]:
    t0 = time.time()
    search = GridSearchCV(build(list(Xtr.columns)), GRID, scoring="average_precision", cv=CV, n_jobs=1)
    search.fit(Xtr, y_train)
    i = search.best_index_
    folds = np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(5)])

    # Secondary metrics: F1-optimal threshold on out-of-fold predictions
    oof = cross_val_predict(search.best_estimator_, Xtr, y_train, cv=CV, method="predict_proba")[:, 1]
    p, r, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    t = float(thr[np.argmax(f1)])
    pred = (oof >= t).astype(int)

    # Top features by gain, mapped back to names
    best = search.best_estimator_
    names = best.named_steps["prep"].get_feature_names_out()
    gain = best.named_steps["model"].get_booster().get_score(importance_type="gain")
    imp = {names[int(k[1:])].split("__")[1]: v for k, v in gain.items()}
    total = sum(imp.values())
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]

    grid_curve = sorted(
        zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]), key=lambda t: -t[1]
    )[:5]
    out[label] = {
        "best_params": {k.split("__")[1]: v for k, v in search.best_params_.items()},
        "fold_ap": folds.round(4).tolist(),
        "mean": round(folds.mean(), 4),
        "sd": round(folds.std(ddof=1), 4),
        "se": round(folds.std(ddof=1) / np.sqrt(5), 4),
        "threshold": round(t, 3),
        "oof_accuracy": round(accuracy_score(y_train, pred), 3),
        "oof_precision": round(precision_score(y_train, pred), 3),
        "oof_recall": round(recall_score(y_train, pred), 3),
        "oof_f1": round(f1_score(y_train, pred), 3),
        "top_gain_share": [(n, round(v / total, 3)) for n, v in top],
        "top5_grid": [({k.split("__")[1]: v for k, v in pp.items()}, round(s, 4)) for pp, s in grid_curve],
        "seconds": round(time.time() - t0, 1),
    }
    print(label, json.dumps(out[label], indent=1))

json.dump(out, open(sys.argv[2], "w"), indent=1)
