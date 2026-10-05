import numpy as np

from src.simulate_reorder import simulate_item


def test_order_arrives_after_lead_time_and_lost_sales_are_not_backordered():
    demand = np.array([5, 5, 5, 5, 5], float)
    # only the decision before day 0 orders: level 10 against 5 on hand -> order 5, which is
    # placed at the close of day -1 and so is on the shelf when day 1 opens (lead time 2)
    levels = np.array([10, np.nan, np.nan, np.nan, np.nan, np.nan])
    sold, lost, stock = simulate_item(demand, levels, start_stock=5, lead_time=2)
    assert sold.tolist() == [5, 5, 0, 0, 0]
    assert lost.tolist() == [0, 0, 5, 5, 5]
    assert stock.tolist() == [0, 0, 0, 0, 0]


def test_no_stockouts_with_enough_stock_and_orders_balance_demand():
    demand = np.full(30, 4.0)
    levels = np.full(31, 8.0)  # two days of demand, lead time 2
    sold, lost, _ = simulate_item(demand, levels, start_stock=8, lead_time=2)
    assert lost.sum() == 0 and sold.sum() == demand.sum()


def test_order_up_to_counts_stock_already_on_order():
    demand = np.zeros(6)
    levels = np.full(7, 10.0)
    _, _, stock = simulate_item(demand, levels, start_stock=0, lead_time=2)
    # one order of 10 is placed up front; later reviews see it in the pipeline and add nothing
    assert stock.max() == 10 and stock[-1] == 10


def test_inventory_never_negative_and_sales_bounded_by_demand():
    rng = np.random.default_rng(0)
    demand = rng.poisson(3, 60).astype(float)
    levels = rng.integers(0, 12, 61).astype(float)
    sold, lost, stock = simulate_item(demand, levels, start_stock=4, lead_time=2)
    assert (stock >= 0).all() and (sold <= demand).all() and np.allclose(sold + lost, demand)
