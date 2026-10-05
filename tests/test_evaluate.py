import numpy as np
import pandas as pd

from src import config
from src.evaluate import bias, mae, make_windows, split, wape


def test_wape_hand_computed():
    y, f = [0, 4, 6], [1, 3, 8]
    assert wape(y, f) == (1 + 1 + 2) / 10


def test_wape_defined_when_some_actuals_are_zero():
    assert np.isfinite(wape([0, 0, 5], [1, 1, 1]))


def test_bias_sign():
    assert bias([5, 5], [6, 6]) > 0
    assert bias([5, 5], [4, 4]) < 0
    assert bias([5, 5], [4, 6]) == 0


def test_mae():
    assert mae([1, 2], [2, 4]) == 1.5


def test_windows_are_ordered_and_do_not_overlap():
    folds, test = make_windows("2016-05-22")
    assert len(folds) == config.N_FOLDS
    edges = folds + [test]
    for (s1, e1), (s2, e2) in zip(edges, edges[1:]):
        assert e1 < s2
    assert (test[1] - test[0]).days + 1 == config.TEST_DAYS


def test_training_rows_end_before_window():
    dates = pd.date_range("2020-01-01", periods=60)
    feats = pd.DataFrame({"target_date": dates, "y": 1})
    window = (dates[30], dates[40])
    train, score = split(feats, window)
    assert train["target_date"].max() < window[0]
    assert score["target_date"].min() >= window[0] and score["target_date"].max() <= window[1]
