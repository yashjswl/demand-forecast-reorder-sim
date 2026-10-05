# Assumptions

## Data and slice

Source is M5 Forecasting (Walmart), store CA_1, daily unit sales from 2011-01-29 to 2016-05-22 (1,941 days). It is retail data used in place of dark-store demand.

The 100 SKUs are chosen by a rule applied only to data before the first validation fold. The calibration window is the 365 days ending 2015-11-29, the day before fold 1 starts. An item is eligible if it had its first sale before that window began and sold at least once in it (2,961 items). Within the window an item is intermittent if at least 50% of its days had zero sales. The remaining items are split at the median of mean daily sales into fast (above) and medium (below). Items are then drawn at random within each segment with a fixed seed: 34 fast, 33 medium, 33 intermittent. Segment labels stay fixed for all later analysis, even if an item's behaviour changes after the window.

M5 records zero sales for the days before an item launched. These are not demand zeros, so each series starts at its first sale (24,366 rows dropped). Price is attached after that. A missing price in M5 means the item was not on sale that week; after trimming there were no missing prices, so the missing-price share is 0% and no price rows were dropped. Prices are forward-filled only, never back-filled. The panel has 169,734 rows. Zero-sales share is 45.3% overall: 22.7% for fast, 45.2% for medium, 68.6% for intermittent. 33% of the series are intermittent by construction.

Observed sales are used as demand. They are censored on days when the item was out of stock, which M5 does not tell us. This understates true demand and is not corrected.

## Forecast set-up

Horizon is 7 days. The unit of forecasting is (SKU, origin day, step h from 1 to 7). Features use data up to and including the origin day. One model is trained for all steps, with h as a feature.

Features built from sales: sales on the origin day and the two days before, rolling mean over 7, 14, 28 and 56 days, rolling standard deviation and share of zero days over 28 days, days since last sale, and sales on the same weekday as the target 1 to 4 weeks earlier. Same-weekday lags are always at or before the origin because the step is at most 7 days.

Features known in advance, describing the target day:

- Weekday and month: fixed by the calendar.
- SNAP flag (US food-benefit payout days): set by a published state schedule.
- Event flag: holidays and sporting events are on the calendar ahead of time.
- Price on the target day, and its ratio to the mean price over the 28 days up to the origin: assumes prices are set in advance. M5 gives realised prices, so a surprise price change would not be known at forecast time; that is an assumption, not a measured fact.

Rows are dropped where any feature is missing, which removes the first 56 days of each series. Lags and rolling windows use only the series' own history.

## Validation

Rolling origin with an expanding training set and three folds of 28 target days, followed by a 91-day test period:

| Window | Target days |
|---|---|
| Fold 1 | 2015-11-30 to 2015-12-27 |
| Fold 2 | 2015-12-28 to 2016-01-24 |
| Fold 3 | 2016-01-25 to 2016-02-21 |
| Test | 2016-02-22 to 2016-05-22 |

For a window, the model is fit on rows whose target day is before the window, then scores rows whose target day falls inside it. An origin can lie inside the window (or just before it) because that is real information at forecast time; the model is not refit within the window. Fold 1 contains Christmas, so its errors are not comparable with the other folds. The test period was scored once, after tuning was frozen.

Tuning is a grid of 8 settings (objective: squared error or Tweedie; leaves: 15 or 63; minimum child samples: 50 or 300), with learning rate 0.05, 300 trees, subsample 0.8, column sample 0.8, seed 42. The choice is by mean WAPE over the three folds. Validation scores are therefore slightly optimistic; the test period is the clean number. The P50 and P90 models use the chosen leaves and minimum child samples with the quantile objective.

## Reorder simulation (simulated)

Everything in this section is simulated. No inventory, lead time, supplier or cost data is observed.

- One SKU at a time, daily review, run over the 91 test days.
- Lead time is 2 days: an order placed after close of day t is on the shelf when day t + 2 opens. Chosen as plausible for a replenished store, not measured.
- Starting stock is 3 days of the SKU's mean sales over the 28 days before the test period, rounded up, with nothing in transit. It is the same for every policy.
- Demand is the observed sales on that day. Demand not met from the shelf is lost, not backordered.
- Order-up-to rule: after each close, order enough to bring on-hand plus on-order stock up to a level, rounded up to whole units. Orders are unlimited in size, with no minimum order, shelf life, capacity or cost.
- Levels use only forecasts made at that origin day:
  - Moving average: 2 days x the 28-day moving average.
  - P50: the sum of the P50 forecasts for the next 2 days.
  - P90: the sum of the P90 forecasts for the next 2 days. Adding daily P90s overstates the P90 of the two-day total, so this policy is conservative by construction.
- When the forecast for the next 2 days is not available (the last days of the window), no order is placed. Those orders would arrive after the window ends.
- To compare policies at equal stock, the level is also scaled by a multiplier from 0.6 to 2.0, and the fill rate is read off at matching average inventory.
