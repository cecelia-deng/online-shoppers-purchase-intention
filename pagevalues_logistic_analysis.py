"""Reproducible logistic-regression analysis for the course project.

This script audits the UCI Online Shoppers Purchasing Intention data,
creates the two EDA figures, and fits L1 logistic regression with and
without PageValues. The held-out test evaluation is disabled by default.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
C_GRID = np.logspace(-3, 2, 11)
DEFAULT_BOOTSTRAP_REPETITIONS = 2_000

CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]

EXPECTED_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("online_shoppers_intention.csv"),
        help="Path to the UCI CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory for figures and result tables.",
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Score the held-out test set. Use once, after all model choices are frozen.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
        help="Number of paired bootstrap samples for the final test AP gap.",
    )
    return parser.parse_args()


def binary_as_int(series: pd.Series, name: str) -> pd.Series:
    """Convert Boolean or Boolean-like values to 0/1 with validation."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        if not set(values.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{name} contains values other than 0 and 1.")
        return values.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    converted = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0})
    if converted.isna().any():
        bad = sorted(normalized[converted.isna()].unique())
        raise ValueError(f"Could not convert {name} to 0/1. Unexpected values: {bad}")
    return converted.astype(int)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the local CSV, falling back to the official UCI Python client."""
    if data_path.exists():
        df = pd.read_csv(data_path)
        source = str(data_path.resolve())
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise FileNotFoundError(
                f"{data_path} was not found. Download the UCI CSV or install ucimlrepo."
            ) from exc
        dataset = fetch_ucirepo(id=468)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        source = "UCI dataset 468 via ucimlrepo"

    missing_columns = sorted(set(EXPECTED_FEATURES + ["Revenue"]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if len(df) != 12_330:
        raise ValueError(f"Expected 12,330 sessions, found {len(df):,}.")
    if df[EXPECTED_FEATURES + ["Revenue"]].isna().any().any():
        raise ValueError("The input contains missing values; the original UCI data should not.")

    df = df[EXPECTED_FEATURES + ["Revenue"]].copy()
    df["Weekend"] = binary_as_int(df["Weekend"], "Weekend")
    df["Revenue"] = binary_as_int(df["Revenue"], "Revenue")
    print(f"Loaded {len(df):,} sessions from {source}")
    return df


def split_data(df: pd.DataFrame):
    """Create the single stratified 80/20 split used by the project."""
    X = df.drop(columns="Revenue")
    y = df["Revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X, y, X_train, X_test, y_train, y_test


def count_test_rows_matching_train(X_train: pd.DataFrame, X_test: pd.DataFrame) -> int:
    """Count test inputs that are exactly equal to at least one training input."""
    train_rows = set(X_train.astype(object).itertuples(index=False, name=None))
    return sum(
        row in train_rows
        for row in X_test.astype(object).itertuples(index=False, name=None)
    )


def audit_data(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tables_dir: Path,
) -> None:
    """Save an auditable table of the dataset and split facts used in the paper."""
    duplicate_mask_all = X.duplicated(keep=False)
    duplicate_group_df = df.loc[duplicate_mask_all]
    page_count = duplicate_group_df[
        ["Administrative", "Informational", "ProductRelated"]
    ].sum(axis=1)
    total_duration = duplicate_group_df[
        ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
    ].sum(axis=1)

    summary = {
        "sessions": len(df),
        "input_features": X.shape[1],
        "purchases": int(y.sum()),
        "purchase_rate": float(y.mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows_beyond_first": int(X.duplicated().sum()),
        "rows_in_duplicate_groups": int(duplicate_mask_all.sum()),
        "purchases_among_duplicate_group_rows": int(duplicate_group_df["Revenue"].sum()),
        "duplicate_group_rows_with_1_or_2_pages": int(page_count.isin([1, 2]).sum()),
        "duplicate_group_rows_with_zero_total_duration": int(total_duration.eq(0).sum()),
        "train_sessions": len(X_train),
        "train_purchase_rate": float(y_train.mean()),
        "test_sessions": len(X_test),
        "test_purchase_rate": float(y_test.mean()),
        "test_input_rows_matching_train": count_test_rows_matching_train(X_train, X_test),
    }
    pd.DataFrame(
        [{"statistic": key, "value": value} for key, value in summary.items()]
    ).to_csv(tables_dir / "data_audit.csv", index=False)

    print("\nData audit")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def create_eda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Create training-only EDA tables and the two figures used in the paper."""
    train = X_train.copy()
    train["Revenue"] = y_train

    monthly = (
        train.groupby("Month", observed=True)["Revenue"]
        .agg(sessions="size", purchases="sum", purchase_rate="mean")
        .reindex(MONTH_ORDER)
        .dropna()
        .reset_index()
    )
    monthly.to_csv(tables_dir / "eda_by_month_training.csv", index=False)

    pagevalues_group = np.where(train["PageValues"].eq(0), "PageValues = 0", "PageValues > 0")
    pagevalues = (
        train.assign(pagevalues_group=pagevalues_group)
        .groupby("pagevalues_group", observed=True)["Revenue"]
        .agg(sessions="size", purchases="sum", purchase_rate="mean")
        .reindex(["PageValues = 0", "PageValues > 0"])
        .reset_index()
    )
    pagevalues["session_share"] = pagevalues["sessions"] / len(train)
    pagevalues.to_csv(tables_dir / "eda_by_pagevalues_training.csv", index=False)

    # Figure 1: separate panels avoid a misleading dual y-axis.
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, layout="constrained")
    x = np.arange(len(monthly))
    axes[0].bar(x, monthly["sessions"], color="#4C78A8")
    axes[0].set_ylabel("Training sessions")
    axes[0].set_title("Session volume and purchase rate by month (training set)", loc="left")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(x, monthly["purchase_rate"], marker="o", linewidth=2, color="#E45756")
    axes[1].set_ylabel("Purchase rate")
    axes[1].set_ylim(0, max(0.30, monthly["purchase_rate"].max() * 1.15))
    axes[1].set_xticks(x, monthly["Month"])
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(figures_dir / "figure_1_monthly_eda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: PageValues split and the corresponding purchase rates.
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout="constrained")
    bars = ax.bar(
        pagevalues["pagevalues_group"],
        pagevalues["purchase_rate"],
        color=["#72B7B2", "#F58518"],
        width=0.62,
    )
    ax.set_ylabel("Purchase rate")
    ax.set_ylim(0, 0.66)
    ax.set_title("Purchase rate by PageValues group (training set)", loc="left")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, rate, sessions in zip(bars, pagevalues["purchase_rate"], pagevalues["sessions"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.018,
            f"{rate:.1%}\n(n={int(sessions):,})",
            ha="center",
            va="bottom",
        )
    fig.savefig(figures_dir / "figure_2_pagevalues_eda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nTraining-set monthly EDA")
    print(monthly.to_string(index=False))
    print("\nTraining-set PageValues EDA")
    print(pagevalues.to_string(index=False))


def make_l1_logistic(random_state: int = RANDOM_STATE) -> LogisticRegression:
    """Construct L1 logistic regression across old and new sklearn APIs."""
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    common = {"solver": "liblinear", "max_iter": 5_000, "random_state": random_state}
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1, **common)
    return LogisticRegression(penalty="l1", **common)


def build_pipeline(columns: list[str]) -> Pipeline:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in columns]
    numeric = [column for column in columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", make_l1_logistic()),
        ]
    )


def run_logistic_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
    cv: StratifiedKFold,
) -> tuple[GridSearchCV, dict, np.ndarray]:
    """Tune C by mean five-fold AP and return the fitted search and fold scores."""
    search = GridSearchCV(
        estimator=build_pipeline(list(X_train.columns)),
        param_grid={"model__C": C_GRID},
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_index = search.best_index_
    fold_ap = np.array(
        [
            search.cv_results_[f"split{fold}_test_score"][best_index]
            for fold in range(cv.n_splits)
        ]
    )
    coefficients = search.best_estimator_.named_steps["model"].coef_.ravel()
    row = {
        "model": label,
        "best_C": search.best_params_["model__C"],
        "features_kept": int(np.count_nonzero(coefficients)),
        "encoded_features": len(coefficients),
        "cv_ap_mean": float(fold_ap.mean()),
        "cv_ap_sd": float(fold_ap.std(ddof=1)),
        "cv_ap_se": float(fold_ap.std(ddof=1) / np.sqrt(len(fold_ap))),
        "cv_ap_min": float(fold_ap.min()),
        "cv_ap_max": float(fold_ap.max()),
    }
    return search, row, fold_ap


def select_f1_threshold(
    fitted_search: GridSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
    cv: StratifiedKFold,
) -> tuple[float, dict]:
    """Choose a decision threshold from training-only out-of-fold probabilities."""
    probabilities = cross_val_predict(
        fitted_search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_train, probabilities)
    f1_curve = (
        2
        * precision_curve[:-1]
        * recall_curve[:-1]
        / np.clip(precision_curve[:-1] + recall_curve[:-1], 1e-12, None)
    )
    best_threshold = float(thresholds[int(np.argmax(f1_curve))])
    predictions = (probabilities >= best_threshold).astype(int)
    row = {
        "model": label,
        "threshold_selected_on_training_oof": best_threshold,
        "oof_accuracy": float(accuracy_score(y_train, predictions)),
        "oof_precision": float(precision_score(y_train, predictions)),
        "oof_recall": float(recall_score(y_train, predictions)),
        "oof_f1": float(f1_score(y_train, predictions)),
    }
    return best_threshold, row


def save_nonzero_coefficients(
    fitted_search: GridSearchCV,
    label: str,
    tables_dir: Path,
) -> None:
    """Save the selected encoded features and fitted coefficients."""
    estimator = fitted_search.best_estimator_
    feature_names = estimator.named_steps["preprocessing"].get_feature_names_out()
    coefficients = estimator.named_steps["model"].coef_.ravel()
    coefficient_table = pd.DataFrame(
        {"model": label, "encoded_feature": feature_names, "coefficient": coefficients}
    )
    coefficient_table = coefficient_table.loc[coefficient_table["coefficient"].ne(0)].copy()
    coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
    coefficient_table = coefficient_table.sort_values("absolute_coefficient", ascending=False)
    safe_label = label.lower().replace(" ", "_").replace("pagevalues", "pagevalues")
    coefficient_table.to_csv(
        tables_dir / f"nonzero_coefficients_{safe_label}.csv",
        index=False,
    )


def plot_logistic_cv(
    fold_with: np.ndarray,
    fold_without: np.ndarray,
    purchase_rate: float,
    figures_dir: Path,
) -> None:
    """Plot fold-level and mean CV AP for both logistic models."""
    labels = ["With PageValues", "Without PageValues"]
    folds = [fold_with, fold_without]
    means = np.array([values.mean() for values in folds])
    ses = np.array([values.std(ddof=1) / np.sqrt(len(values)) for values in folds])
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 3.5), layout="constrained")
    ax.barh(y, means, height=0.50, color="#4C78A8", alpha=0.90)
    ax.errorbar(means, y, xerr=ses, fmt="none", color="black", capsize=4)
    for y_value, fold_values in zip(y, folds):
        ax.scatter(fold_values, np.full(len(fold_values), y_value + 0.33), color="#555555", s=20)
    for y_value, mean, se in zip(y, means, ses):
        ax.text(mean + se + 0.015, y_value, f"{mean:.3f} ± {se:.3f}", va="center")
    ax.axvline(purchase_rate, color="#777777", linestyle="--", linewidth=1.2)
    ax.text(
        purchase_rate + 0.010,
        0.50,
        f"Purchase-rate floor = {purchase_rate:.3f}",
        rotation=90,
        color="#666666",
        fontsize=9,
        ha="left",
        va="center",
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Average precision (stratified 5-fold CV on training set)")
    ax.set_title("L1 logistic regression cross-validation", loc="left")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(figures_dir / "figure_3_logistic_cv_ap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_bootstrap_ap_gap(
    y_true: pd.Series,
    score_with: np.ndarray,
    score_without: np.ndarray,
    repetitions: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, float, float]:
    """Bootstrap the paired AP(with) - AP(without) difference on test sessions."""
    rng = np.random.default_rng(seed)
    y_array = np.asarray(y_true)
    with_array = np.asarray(score_with)
    without_array = np.asarray(score_without)
    gaps = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        if y_array[indices].sum() == 0:
            continue
        gap = average_precision_score(y_array[indices], with_array[indices]) - average_precision_score(
            y_array[indices], without_array[indices]
        )
        gaps.append(gap)
    gaps_array = np.asarray(gaps)
    lower, upper = np.percentile(gaps_array, [2.5, 97.5])
    return gaps_array, float(lower), float(upper)


def run_final_test(
    search_with: GridSearchCV,
    search_without: GridSearchCV,
    threshold_with: float,
    threshold_without: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    bootstrap_repetitions: int,
    tables_dir: Path,
) -> None:
    """Perform the one-time held-out evaluation and paired bootstrap."""
    probability_with = search_with.predict_proba(X_test)[:, 1]
    probability_without = search_without.predict_proba(X_test.drop(columns="PageValues"))[:, 1]
    predictions_with = (probability_with >= threshold_with).astype(int)
    predictions_without = (probability_without >= threshold_without).astype(int)

    ap_with = float(average_precision_score(y_test, probability_with))
    ap_without = float(average_precision_score(y_test, probability_without))
    gap_samples, ci_lower, ci_upper = paired_bootstrap_ap_gap(
        y_test,
        probability_with,
        probability_without,
        repetitions=bootstrap_repetitions,
    )

    rows = []
    for label, probabilities, predictions, threshold in [
        ("L1 logistic with PageValues", probability_with, predictions_with, threshold_with),
        ("L1 logistic without PageValues", probability_without, predictions_without, threshold_without),
    ]:
        rows.append(
            {
                "model": label,
                "test_ap": float(average_precision_score(y_test, probabilities)),
                "threshold_selected_on_training_oof": threshold,
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(precision_score(y_test, predictions)),
                "test_recall": float(recall_score(y_test, predictions)),
                "test_f1": float(f1_score(y_test, predictions)),
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "logistic_test_results.csv", index=False)
    pd.DataFrame({"bootstrap_ap_gap_with_minus_without": gap_samples}).to_csv(
        tables_dir / "paired_bootstrap_gap_samples.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "test_ap_with_pagevalues": ap_with,
                "test_ap_without_pagevalues": ap_without,
                "test_ap_gap_with_minus_without": ap_with - ap_without,
                "paired_bootstrap_ci_lower_95": ci_lower,
                "paired_bootstrap_ci_upper_95": ci_upper,
                "bootstrap_repetitions_requested": bootstrap_repetitions,
                "bootstrap_repetitions_used": len(gap_samples),
                "bootstrap_seed": RANDOM_STATE,
            }
        ]
    ).to_csv(tables_dir / "paired_bootstrap_summary.csv", index=False)

    print("\nFINAL HELD-OUT TEST RESULTS")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"AP gap (with - without): {ap_with - ap_without:.6f}; "
        f"95% paired bootstrap CI [{ci_lower:.6f}, {ci_upper:.6f}]"
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive.")

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X, y, X_train, X_test, y_train, y_test = split_data(df)
    audit_data(df, X, y, X_train, X_test, y_train, y_test, tables_dir)
    create_eda(X_train, y_train, figures_dir, tables_dir)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search_with, row_with, fold_with = run_logistic_cv(
        X_train,
        y_train,
        label="L1 logistic with PageValues",
        cv=cv,
    )
    search_without, row_without, fold_without = run_logistic_cv(
        X_train.drop(columns="PageValues"),
        y_train,
        label="L1 logistic without PageValues",
        cv=cv,
    )

    cv_results = pd.DataFrame(
        [
            {
                "model": "Constant-score reference",
                "cv_ap_mean": float(y_train.mean()),
            },
            row_with,
            row_without,
        ]
    )
    cv_results.to_csv(tables_dir / "logistic_cv_results.csv", index=False)
    fold_table = pd.DataFrame(
        {
            "fold": np.arange(1, N_SPLITS + 1),
            "ap_with_pagevalues": fold_with,
            "ap_without_pagevalues": fold_without,
        }
    )
    fold_table.to_csv(tables_dir / "logistic_cv_fold_scores.csv", index=False)

    threshold_with, secondary_with = select_f1_threshold(
        search_with, X_train, y_train, "L1 logistic with PageValues", cv
    )
    threshold_without, secondary_without = select_f1_threshold(
        search_without,
        X_train.drop(columns="PageValues"),
        y_train,
        "L1 logistic without PageValues",
        cv,
    )
    secondary = pd.DataFrame(
        [
            {
                "model": 'Always predict "no purchase"',
                "oof_accuracy": float(1 - y_train.mean()),
                "oof_precision": 0.0,
                "oof_recall": 0.0,
                "oof_f1": 0.0,
            },
            secondary_with,
            secondary_without,
        ]
    )
    secondary.to_csv(tables_dir / "logistic_oof_secondary_metrics.csv", index=False)

    save_nonzero_coefficients(search_with, "with PageValues", tables_dir)
    save_nonzero_coefficients(search_without, "without PageValues", tables_dir)
    plot_logistic_cv(fold_with, fold_without, float(y_train.mean()), figures_dir)

    manifest = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": N_SPLITS,
        "C_grid": C_GRID.tolist(),
        "final_test_run": bool(args.run_final_test),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nLogistic regression CV results")
    print(cv_results.to_string(index=False))
    print("\nTraining-only out-of-fold secondary metrics")
    print(secondary.to_string(index=False))

    if args.run_final_test:
        run_final_test(
            search_with,
            search_without,
            threshold_with,
            threshold_without,
            X_test,
            y_test,
            args.bootstrap_repetitions,
            tables_dir,
        )
    else:
        print(
            "\nHeld-out test scoring was NOT run. "
            "Use --run-final-test once, after every model choice is frozen."
        )

    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
