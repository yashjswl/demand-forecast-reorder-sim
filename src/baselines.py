from pathlib import Path

import pandas as pd

from src import config
from src.evaluate import make_windows, split, summarize
from src.features import build_features

# Each baseline reads only columns that are known at the origin day.
BASELINES = {
    "naive": "lag_0",             # last observed day
    "seasonal_naive_7": "wd_lag_1",  # same weekday as the target, one week earlier
    "ma_28": "roll_mean_28",
}


def add_baselines(feats):
    out = feats.copy()
    for name, col in BASELINES.items():
        out[name] = out[col]
    return out


def main():
    panel = pd.read_parquet("data/processed/panel.parquet")
    feats = add_baselines(build_features(panel))
    folds, _ = make_windows(panel["date"].max())
    results = []
    for k, window in enumerate(folds, 1):
        _, score = split(feats, window)
        res = summarize(score, list(BASELINES))
        res["fold"] = k
        results.append(res)
    res = pd.concat(results)
    Path("reports").mkdir(exist_ok=True)
    res.to_csv("reports/baselines_validation.csv", index=False)

    for k, w in enumerate(folds, 1):
        print(f"fold {k}: targets {w[0].date()} to {w[1].date()}")
    pd.options.display.float_format = "{:.3f}".format
    print(res.pivot_table(index=["segment", "model"], columns="fold", values="wape"))
    print(res.groupby(["segment", "model"])[["wape", "bias", "mae"]].mean())


if __name__ == "__main__":
    main()
