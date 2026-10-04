"""
Shared helpers used by every notebook, so each notebook prepares the data
and scores the models in exactly the same way (a fair comparison).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "uk_card_transactions.csv.gz"
TARGET = "is_fraud"

# Business assumptions used to put a GBP value on mistakes (made up, but reasonable):
FALSE_ALARM_COST_GBP = 5.0   # cost of blocking a genuine payment: call centre time, annoyed customer
# A missed fraud costs the bank the full transaction amount (refund under UK rules).

NUMERIC_FEATURES = [
    "amount_gbp", "log_amount", "amount_to_avg_ratio", "distance_from_home_km",
    "account_age_months", "txn_count_last_1h", "txn_count_last_24h",
    "failed_pin_attempts", "hour", "is_night", "is_weekend", "is_international",
    "is_new_merchant", "sca_authenticated", "region_mismatch",
]
CATEGORICAL_FEATURES = ["merchant_category", "channel", "card_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    """Read the CSV and add the engineered features."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return add_features(df)


def add_features(df):
    """Feature engineering - turn raw columns into signals a model can use."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df["log_amount"] = np.log1p(df["amount_gbp"])          # amounts are very skewed
    df["amount_to_avg_ratio"] = df["amount_gbp"] / df["customer_avg_spend_gbp"]
    df["region_mismatch"] = (df["transaction_region"] != df["home_region"]).astype(int)
    return df


def time_split(df):
    """
    Split by DATE, not at random:
        train      = 1-21 Sep  (learn)
        validation = 22-25 Sep (tune settings / choose threshold)
        test       = 26-30 Sep (final score, touched once)
    This copies real life: a bank trains on the past and predicts the future.
    """
    day = df["timestamp"].dt.day
    train = df[day <= 21]
    val = df[(day > 21) & (day <= 25)]
    test = df[day > 25]
    return train, val, test


def make_preprocessor(scale=True):
    """Scale numbers and one-hot encode categories. Fitted on TRAIN data only."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def business_cost(y_true, y_pred, amounts):
    """Total GBP cost = value of fraud we missed + GBP 5 per false alarm."""
    y_true, y_pred, amounts = map(np.asarray, (y_true, y_pred, amounts))
    missed = amounts[(y_true == 1) & (y_pred == 0)].sum()
    false_alarms = ((y_true == 0) & (y_pred == 1)).sum()
    return missed + false_alarms * FALSE_ALARM_COST_GBP


def best_threshold(y_true, scores, amounts, grid=None):
    """Pick the probability cut-off that gives the lowest GBP cost (use on VALIDATION data)."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
    costs = [business_cost(y_true, (scores >= t).astype(int), amounts) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(name, y_true, y_pred, scores=None, amounts=None, show_plot=True):
    """Print the metrics that matter for fraud and return them as a dict."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    res = {
        "model": name,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall (fraud caught)": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else np.nan,
        "frauds_missed": int(fn),
        "false_alarms": int(fp),
    }
    if scores is not None:
        res["pr_auc"] = average_precision_score(y_true, scores)
        res["roc_auc"] = roc_auc_score(y_true, scores)
    if amounts is not None:
        res["cost_gbp"] = business_cost(y_true, y_pred, amounts)

    print(f"--- {name} ---")
    for k, v in res.items():
        if k != "model":
            print(f"{k:>24}: {v:,.4f}" if isinstance(v, float) else f"{k:>24}: {v:,}")
    if show_plot:
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                               display_labels=["Genuine", "Fraud"]).plot(cmap="Blues", values_format=",")
        plt.title(name)
        plt.show()
    return res
