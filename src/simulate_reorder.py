from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.evaluate import make_windows

REPORTS = Path("reports")
P = config.LEAD_TIME  # review every day, so the protection period equals the lead time


def simulate_item(demand, order_up_to, start_stock, lead_time):
    """SIMULATED single-SKU inventory run with daily review and lost sales.

    order_up_to[0] is decided before day 0 opens, order_up_to[i + 1] at the close of day i.
    An order placed at close of day i is available when day i + lead_time opens.
    NaN means no forecast was available, so no order is placed.
    """
    n = len(demand)
    onhand = float(start_stock)
    pipeline = {}
    sold = np.zeros(n)
    stock_end = np.zeros(n)

    def review(slot, day_index):
        level = order_up_to[slot]
        if np.isnan(level):
            return
        position = onhand + sum(pipeline.values())
        qty = max(0.0, np.ceil(level) - position)
        if qty > 0:
            arrival = day_index + lead_time
            pipeline[arrival] = pipeline.get(arrival, 0.0) + qty

    review(0, -1)
    for i in range(n):
        onhand += pipeline.pop(i, 0.0)
        sold[i] = min(onhand, demand[i])
        onhand -= sold[i]
        stock_end[i] = onhand
        review(i + 1, i)
    return sold, np.asarray(demand, float) - sold, stock_end


def levels_from_forecasts(item_preds, test_start, n_days, policy, k):
    """Order-up-to level for each decision point, from forecasts made at that origin only."""
    out = np.full(n_days + 1, np.nan)
    origins = pd.date_range(test_start - pd.Timedelta(days=1), periods=n_days + 1)
    wide = item_preds.pivot(index="origin", columns="h", values=policy)
    for slot, origin in enumerate(origins):
        if origin in wide.index and all(h in wide.columns and not np.isnan(wide.at[origin, h]) for h in range(1, P + 1)):
            if policy == "ma_28":
                out[slot] = k * P * wide.at[origin, 1]
            else:
                out[slot] = k * sum(wide.at[origin, h] for h in range(1, P + 1))
    return out


def run(preds, panel, test_window, policy, k):
    start, end = test_window
    n_days = (end - start).days + 1
    rows = []
    for item, g in preds.groupby("item_id"):
        s = panel[panel["item_id"] == item].set_index("date")["sales"]
        demand = s.loc[start:end].to_numpy(float)
        start_stock = np.ceil(config.START_COVER_DAYS * s.loc[: start - pd.Timedelta(days=1)].tail(28).mean())
        levels = levels_from_forecasts(g, start, n_days, policy, k)
        sold, lost, stock = simulate_item(demand, levels, start_stock, config.LEAD_TIME)
        rows.append({"item_id": item, "segment": g["segment"].iloc[0], "demand": demand.sum(), "sold": sold.sum(),
                     "stockout_days": int((lost > 0).sum()), "days": n_days, "inventory_sum": stock.sum()})
    return pd.DataFrame(rows)


def summarize(per_item):
    out = []
    for seg, g in [("all", per_item)] + list(per_item.groupby("segment")):
        out.append({"segment": seg, "fill_rate": g["sold"].sum() / g["demand"].sum(),
                    "stockout_day_share": g["stockout_days"].sum() / g["days"].sum(),
                    "avg_inventory_units": g["inventory_sum"].sum() / g["days"].sum()})
    return pd.DataFrame(out)


def main():
    panel = pd.read_parquet("data/processed/panel.parquet")
    preds = pd.read_parquet(REPORTS / "test_predictions.parquet")
    _, test_window = make_windows(panel["date"].max())
    policies = {"moving_average": "ma_28", "p50": "q50", "p90": "q90"}
    base, sweep = [], []
    for name, col in policies.items():
        for k in config.SWEEP:
            res = summarize(run(preds, panel, test_window, col, k))
            res.insert(0, "policy", name)
            res.insert(1, "multiplier", k)
            sweep.append(res)
            if k == 1.0:
                base.append(res)
    pd.concat(base).to_csv(REPORTS / "simulation_results.csv", index=False)
    pd.concat(sweep).to_csv(REPORTS / "simulation_sweep.csv", index=False)
    pd.options.display.float_format = "{:.3f}".format
    print(pd.concat(base).drop(columns="multiplier").to_string(index=False))


if __name__ == "__main__":
    main()
