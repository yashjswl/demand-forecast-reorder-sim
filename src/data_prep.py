import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

RAW = Path("data/raw")
OUT = Path("data/processed")

EVENT_COLS = ["event_name_1", "event_type_1"]


def load_raw(store):
    sales_file = next(
        (RAW / f for f in ["sales_train_evaluation.csv", "sales_train_validation.csv"] if (RAW / f).exists()),
        None,
    )
    if sales_file is None:
        raise FileNotFoundError("put M5 sales_train_*.csv, calendar.csv, sell_prices.csv in data/raw/")
    sales = pd.read_csv(sales_file)
    sales = sales[sales["store_id"] == store].reset_index(drop=True)
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"] == store]
    return sales, calendar, prices


def to_long(sales, calendar):
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=["item_id"], value_vars=day_cols, var_name="d", value_name="sales")
    long = long.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")
    return long.drop(columns="d").sort_values(["item_id", "date"]).reset_index(drop=True)


def choose_slice(long, n_days, rng):
    """Pick N_ITEMS SKUs, a third each from fast, medium and intermittent."""
    dates = np.sort(long["date"].unique())
    fold1_start = n_days - config.TEST_DAYS - config.N_FOLDS * config.FOLD_DAYS
    cal_end = dates[fold1_start - 1]
    cal_start = dates[fold1_start - config.CALIBRATION_DAYS]

    first_sale = long[long["sales"] > 0].groupby("item_id")["date"].min()
    cal = long[(long["date"] >= cal_start) & (long["date"] <= cal_end)]
    stats = cal.groupby("item_id")["sales"].agg(mean_sales="mean", zero_share=lambda s: (s == 0).mean())
    # need a full calibration year of history, otherwise zeros are "not yet launched"
    stats = stats[first_sale.reindex(stats.index) <= cal_start]
    stats = stats[stats["mean_sales"] > 0]

    stats["segment"] = "intermittent"
    regular = stats["zero_share"] < config.INTERMITTENT_ZERO_SHARE
    cut = stats.loc[regular, "mean_sales"].quantile(0.5)
    stats.loc[regular & (stats["mean_sales"] >= cut), "segment"] = "fast"
    stats.loc[regular & (stats["mean_sales"] < cut), "segment"] = "medium"

    per_seg = [config.N_ITEMS // 3 + (i < config.N_ITEMS % 3) for i in range(3)]
    picks = []
    for seg, k in zip(["fast", "medium", "intermittent"], per_seg):
        pool = stats[stats["segment"] == seg]
        picks.append(pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(1 << 31))))
    chosen = pd.concat(picks)
    return chosen, stats, cal_end


def attach_prices(long, prices, calendar):
    long = long.merge(prices[["item_id", "wm_yr_wk", "sell_price"]], on=["item_id", "wm_yr_wk"], how="left")
    before = long["sell_price"].isna().mean()
    # forward fill only: back-filling would carry a later price into the past
    long["sell_price"] = long.groupby("item_id")["sell_price"].ffill()
    return long, before


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    sales, calendar, prices = load_raw(config.STORE)
    long = to_long(sales, calendar)
    n_days = long["date"].nunique()

    chosen, pool_stats, cal_end = choose_slice(long, n_days, rng)
    panel = long[long["item_id"].isin(chosen.index)].copy()

    # M5 lists zeros for days before an item launched; those are not demand zeros.
    first_sale = panel[panel["sales"] > 0].groupby("item_id")["date"].min()
    leading = panel["date"] < panel["item_id"].map(first_sale)
    n_leading = int(leading.sum())
    panel = panel[~leading]

    panel, missing_price_share = attach_prices(panel, prices, calendar)
    still_missing = int(panel["sell_price"].isna().sum())
    panel = panel.dropna(subset=["sell_price"])

    cal_cols = ["date", "wday", "month", "snap_CA", "event_name_1", "event_type_1"]
    panel = panel.merge(calendar[cal_cols], on="date", how="left")
    panel["has_event"] = panel["event_name_1"].notna().astype(int)
    panel = panel.drop(columns=["wm_yr_wk", "event_name_1", "event_type_1"])
    panel = panel.rename(columns={"snap_CA": "snap"})

    seg_map = chosen["segment"].rename("segment").reset_index()
    panel = panel.merge(seg_map, on="item_id")
    panel = panel.sort_values(["item_id", "date"]).reset_index(drop=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)

    zero_share = panel.groupby("segment")["sales"].apply(lambda s: float((s == 0).mean()))
    summary = {
        "store": config.STORE,
        "n_days_in_data": int(n_days),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "calibration_end": str(pd.Timestamp(cal_end).date()),
        "eligible_items_in_store": int(len(pool_stats)),
        "eligible_by_segment": pool_stats["segment"].value_counts().to_dict(),
        "chosen_by_segment": chosen["segment"].value_counts().to_dict(),
        "n_series": int(panel["item_id"].nunique()),
        "n_rows": int(len(panel)),
        "leading_pre_launch_zero_rows_dropped": n_leading,
        "zero_sales_share_overall": float((panel["sales"] == 0).mean()),
        "zero_sales_share_by_segment": zero_share.to_dict(),
        "intermittent_series_share": float((chosen["segment"] == "intermittent").mean()),
        "price_missing_share_before_ffill": float(missing_price_share),
        "price_rows_dropped_after_ffill": still_missing,
    }
    (OUT / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    chosen.reset_index().to_csv(OUT / "slice.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
