import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, HISTORY, SAME_WEEKDAY, build_features

H = 7


def make_panel(n_days=200, n_items=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days)
    frames = []
    for i in range(n_items):
        frames.append(pd.DataFrame({
            "item_id": f"I{i}",
            "date": dates,
            "sales": rng.poisson(3, n_days),
            "sell_price": 2.0 + rng.random(n_days),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "snap": rng.integers(0, 2, n_days),
            "has_event": rng.integers(0, 2, n_days),
            "segment": "fast",
        }))
    return pd.concat(frames, ignore_index=True)


def test_features_ignore_data_after_cutoff():
    panel = make_panel()
    cutoff = pd.Timestamp("2020-04-15")
    base = build_features(panel, H)

    # scramble everything observed after the cutoff; prices are allowed to be known
    # H days ahead, so they are only touched beyond cutoff + H
    changed = panel.copy()
    after = changed["date"] > cutoff
    changed.loc[after, "sales"] = 10_000
    changed.loc[changed["date"] > cutoff + pd.Timedelta(days=H), "sell_price"] = 999.0
    rebuilt = build_features(changed, H)

    key = ["item_id", "origin", "h"]
    a = base[base["origin"] <= cutoff].set_index(key)
    b = rebuilt[rebuilt["origin"] <= cutoff].set_index(key)
    cols = [c for c in FEATURES if c != "h"]
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_target_is_sales_on_target_date():
    panel = make_panel()
    feats = build_features(panel, H)
    truth = panel.rename(columns={"date": "target_date", "sales": "y_true"})[["item_id", "target_date", "y_true"]]
    merged = feats.merge(truth, on=["item_id", "target_date"])
    assert len(merged) == len(feats)
    assert (merged["y"] == merged["y_true"]).all()


@pytest.mark.parametrize("h", range(1, H + 1))
def test_same_weekday_lag_is_not_in_the_future(h):
    panel = make_panel(seed=1)
    feats = build_features(panel, H)
    f = feats[(feats["h"] == h) & (feats["item_id"] == "I0")].iloc[10]
    s = panel[panel["item_id"] == "I0"].set_index("date")["sales"]
    for j in range(1, 5):
        src_date = f["target_date"] - pd.Timedelta(days=7 * j)
        assert src_date <= f["origin"]
        assert f[f"wd_lag_{j}"] == s[src_date]


def test_target_matches_origin_plus_horizon():
    feats = build_features(make_panel(), H)
    assert ((feats["target_date"] - feats["origin"]).dt.days == feats["h"]).all()
    assert (feats["target_date"] > feats["origin"]).all()


def test_gaps_are_rejected():
    panel = make_panel()
    panel = panel.drop(panel.index[50])
    with pytest.raises(ValueError):
        build_features(panel, H)
