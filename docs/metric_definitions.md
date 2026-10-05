# Metric definitions

All point metrics are pooled: errors and actuals are summed over every SKU, origin day and horizon step in the group (all rows, or one segment) before dividing. Fast SKUs therefore carry more weight than intermittent ones in the "all" rows.

**WAPE** (primary) = sum(|forecast - actual|) / sum(actual). Lower is better. A forecast of zero on every row scores exactly 1.0, so a WAPE above 1.0 means the forecast is worse than predicting no sales on this measure. That happens for the intermittent segment, where most actuals are zero and a forecast that is positive on every day pays for each zero.

**Why not MAPE.** MAPE divides each error by that day's actual. About 45% of SKU-days in this slice have zero sales (see the README), and MAPE is undefined on all of them. Dropping those days would score the model only on days when something sold, which hides exactly the errors that matter for stocking. Replacing zeros with a small constant makes the metric a function of that constant. WAPE divides by total sales, so it is always defined and weights errors by volume.

**Bias** = sum(forecast - actual) / sum(actual). Positive means over-forecasting. Reported as a signed fraction of total sales, so 0.05 is 5% over.

**MAE** = mean(|forecast - actual|), in units per SKU-day. Included because WAPE is hard to read on intermittent SKUs.

**Pinball loss** for a quantile forecast q at level a: mean of max(a * (y - q), (a - 1) * (y - q)). Used for the P50 and P90 models.

**Coverage** = share of actuals at or below a quantile forecast. A well-calibrated P90 forecast covers about 90%. For P50 on intermittent SKUs coverage is well above 50% because the median forecast is zero on most days and actuals are zero on most days.

**Simulation metrics** (simulated, see assumptions.md)

- Fill rate = units sold / units demanded, summed over SKUs and days.
- Stockout-day share = share of SKU-days on which demand exceeded the stock on the shelf.
- Average inventory = mean end-of-day on-hand units per SKU per day.
