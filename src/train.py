import argparse
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.baselines import add_baselines, BASELINES
from src.evaluate import coverage, make_windows, pinball, split, summarize, wape
from src.features import FEATURES, build_features

REPORTS = Path("reports")
BASE = dict(learning_rate=0.05, n_estimators=300, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=config.SEED, deterministic=True,
            force_row_wise=True, verbose=-1)

# Small grid, scored on the validation folds only.
GRID = {
    "objective": ["regression", "tweedie"],
    "num_leaves": [15, 63],
    "min_child_samples": [50, 300],
}


def fit(train, **params):
    model = lgb.LGBMRegressor(**{**BASE, **params})
    model.fit(train[FEATURES], train["y"], categorical_feature=["wday", "month"])
    return model


def predict(model, df):
    return np.clip(model.predict(df[FEATURES]), 0, None)


def tune(feats, folds):
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        scores = []
        for window in folds:
            train, score = split(feats, window)
            scores.append(wape(score["y"], predict(fit(train, **params), score)))
        rows.append({**params, "wape_mean": float(np.mean(scores)),
                     **{f"wape_f{i+1}": s for i, s in enumerate(scores)}})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows).sort_values("wape_mean").reset_index(drop=True)


def fit_all(train, params):
    structural = {k: v for k, v in params.items() if k != "objective"}
    return {
        "lgbm": fit(train, **params),
        "q50": fit(train, objective="quantile", alpha=0.5, **structural),
        "q90": fit(train, objective="quantile", alpha=0.9, **structural),
    }


def score_models(models, df):
    out = df.copy()
    for name, m in models.items():
        out[name] = predict(m, df)
    return out


def quantile_report(df):
    return pd.DataFrame([
        {"segment": s, "q50_pinball": pinball(g["y"], g["q50"], 0.5), "q50_coverage": coverage(g["y"], g["q50"]),
         "q90_pinball": pinball(g["y"], g["q90"], 0.9), "q90_coverage": coverage(g["y"], g["q90"])}
        for s, g in [("all", df)] + list(df.groupby("segment"))
    ])


def validate(feats, folds):
    REPORTS.mkdir(exist_ok=True)
    tuning = tune(feats, folds)
    tuning.to_csv(REPORTS / "tuning_validation.csv", index=False)
    best = {k: tuning.loc[0, k] for k in GRID}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    print("chosen params:", best)
    (REPORTS / "chosen_params.json").write_text(json.dumps(best, indent=2))

    keep = ["item_id", "origin", "target_date", "h", "y", "segment"]
    scored, summaries, qsums = [], [], []
    for k, window in enumerate(folds, 1):
        train, score = split(feats, window)
        models = fit_all(train, best)
        df = score_models(models, score)
        df["fold"] = k
        scored.append(df[keep + ["fold", "lgbm", "q50", "q90"] + list(BASELINES)])
        s = summarize(df, ["lgbm"] + list(BASELINES)); s["fold"] = k; summaries.append(s)
        q = quantile_report(df); q["fold"] = k; qsums.append(q)
    scored = pd.concat(scored)
    scored.to_parquet(REPORTS / "validation_predictions.parquet", index=False)
    pd.concat(summaries).to_csv(REPORTS / "model_validation.csv", index=False)
    pd.concat(qsums).to_csv(REPORTS / "quantile_validation.csv", index=False)

    imp = pd.Series(models["lgbm"].booster_.feature_importance("gain"), index=FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).rename("gain_share").to_csv(REPORTS / "feature_importance_val_fold3.csv")
    return pd.concat(summaries), pd.concat(qsums), scored


def final(feats, test_window, params):
    # Run once, after validation choices are frozen.
    train, score = split(feats, test_window)
    models = fit_all(train, params)
    df = score_models(models, score)
    keep = ["item_id", "origin", "target_date", "h", "y", "segment", "lgbm", "q50", "q90"] + list(BASELINES)
    df[keep].to_parquet(REPORTS / "test_predictions.parquet", index=False)
    summarize(df, ["lgbm"] + list(BASELINES)).to_csv(REPORTS / "model_test.csv", index=False)
    quantile_report(df).to_csv(REPORTS / "quantile_test.csv", index=False)
    imp = pd.Series(models["lgbm"].booster_.feature_importance("gain"), index=FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).rename("gain_share").to_csv(REPORTS / "feature_importance_final.csv")
    print(pd.read_csv(REPORTS / "model_test.csv").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="score the held-out test period (once)")
    args = ap.parse_args()
    panel = pd.read_parquet("data/processed/panel.parquet")
    feats = add_baselines(build_features(panel))
    folds, test_window = make_windows(panel["date"].max())
    if args.final:
        final(feats, test_window, json.loads((REPORTS / "chosen_params.json").read_text()))
        return
    summ, qsum, _ = validate(feats, folds)
    pd.options.display.float_format = "{:.3f}".format
    print(summ.groupby(["segment", "model"])[["wape", "bias", "mae"]].mean())
    print(qsum.groupby("segment").mean(numeric_only=True))


if __name__ == "__main__":
    main()
