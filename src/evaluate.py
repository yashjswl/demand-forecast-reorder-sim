import numpy as np
import pandas as pd

from src import config

SEGMENTS = ["fast", "medium", "intermittent"]


def wape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return np.abs(f - y).sum() / y.sum()


def bias(y, f):
    """Signed: positive means over-forecasting, as a share of total actual sales."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return (f - y).sum() / y.sum()


def mae(y, f):
    return float(np.mean(np.abs(np.asarray(f, float) - np.asarray(y, float))))


def make_windows(last_date):
    """Validation windows (target dates) for the rolling-origin folds, plus the test window."""
    last = pd.Timestamp(last_date)
    test_start = last - pd.Timedelta(days=config.TEST_DAYS - 1)
    folds = []
    for k in range(config.N_FOLDS):
        end = test_start - pd.Timedelta(days=1 + config.FOLD_DAYS * (config.N_FOLDS - 1 - k))
        start = end - pd.Timedelta(days=config.FOLD_DAYS - 1)
        folds.append((start, end))
    return folds, (test_start, last)


def split(feats, window):
    """Rows scored in a window: target day inside it. Rows used to fit: target day before it."""
    start, end = window
    train = feats[feats["target_date"] < start]
    score = feats[(feats["target_date"] >= start) & (feats["target_date"] <= end)]
    return train, score


def summarize(df, pred_cols, by=("segment",)):
    rows = []
    groups = [("all", df)] + [(s, df[df["segment"] == s]) for s in SEGMENTS]
    for name, g in groups:
        for c in pred_cols:
            rows.append({"segment": name, "model": c, "wape": wape(g["y"], g[c]),
                         "bias": bias(g["y"], g[c]), "mae": mae(g["y"], g[c]), "n": len(g)})
    return pd.DataFrame(rows)


def pinball(y, q, alpha):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y, q):
    """Share of actuals at or below the quantile forecast; should be close to alpha."""
    return float(np.mean(np.asarray(y, float) <= np.asarray(q, float)))
