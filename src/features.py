import numpy as np
import pandas as pd

from src import config

# Every feature is a function of data up to and including the origin day, except
# the "known ahead" group, which describes the target day itself.
HISTORY = ["lag_0", "lag_1", "lag_2", "roll_mean_7", "roll_mean_14", "roll_mean_28",
           "roll_mean_56", "roll_std_28", "roll_zero_28", "days_since_sale"]
SAME_WEEKDAY = ["wd_lag_1", "wd_lag_2", "wd_lag_3", "wd_lag_4"]
KNOWN_AHEAD = ["h", "wday", "month", "snap", "has_event", "price", "price_ratio"]
FEATURES = HISTORY + SAME_WEEKDAY + KNOWN_AHEAD


def _days_since_sale(s):
    # s is a daily series; count back from the origin day to the last day with a sale
    idx = np.arange(len(s))
    last = pd.Series(np.where(s.to_numpy() > 0, idx, np.nan), index=s.index).ffill()
    return pd.Series(idx - last.to_numpy(), index=s.index).fillna(len(s))


def _item_features(g, horizon):
    g = g.sort_values("date").reset_index(drop=True)
    if (g["date"].diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError(f"gap in dates for {g['item_id'].iloc[0]}")
    s = g["sales"].astype(float)
    base = pd.DataFrame({"origin": g["date"]})
    for k in (0, 1, 2):
        base[f"lag_{k}"] = s.shift(k)
    for w in (7, 14, 28, 56):
        base[f"roll_mean_{w}"] = s.rolling(w, min_periods=w).mean()
    base["roll_std_28"] = s.rolling(28, min_periods=28).std()
    base["roll_zero_28"] = (s == 0).astype(float).rolling(28, min_periods=28).mean()
    base["days_since_sale"] = _days_since_sale(s)
    # trailing mean price over days up to the origin, used to scale the target-day price
    base["price_ref"] = g["sell_price"].rolling(28, min_periods=1).mean()

    parts = []
    for h in range(1, horizon + 1):
        f = base.copy()
        f["h"] = h
        f["target_date"] = g["date"].shift(-h)
        f["y"] = s.shift(-h)
        # sales on the same weekday as the target, 1..4 weeks before it; all <= origin since h <= 7
        for j in range(1, 5):
            f[f"wd_lag_{j}"] = s.shift(7 * j - h)
        for col in ("wday", "month", "snap", "has_event"):
            f[col] = g[col].shift(-h)
        f["price"] = g["sell_price"].shift(-h)
        f["price_ratio"] = f["price"] / f["price_ref"]
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    out["item_id"] = g["item_id"].iloc[0]
    return out


def build_features(panel, horizon=config.HORIZON):
    """One row per (item, origin day, horizon step). `y` is the sales on `target_date`."""
    assert horizon <= 7, "same-weekday lags assume horizon <= 7"
    out = pd.concat([_item_features(g, horizon) for _, g in panel.groupby("item_id")], ignore_index=True)
    out = out.dropna(subset=["target_date", "y"]).drop(columns="price_ref")
    out = out.dropna(subset=FEATURES)
    for c in ("wday", "month", "snap", "has_event"):
        out[c] = out[c].astype(int)
    seg = panel.drop_duplicates("item_id").set_index("item_id")["segment"]
    out["segment"] = out["item_id"].map(seg)
    return out.sort_values(["item_id", "origin", "h"]).reset_index(drop=True)
