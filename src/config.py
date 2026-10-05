SEED = 42
STORE = "CA_1"
N_ITEMS = 100
HORIZON = 7

# Time layout, counted back from the last day in the data.
TEST_DAYS = 91          # final holdout, scored once
FOLD_DAYS = 28          # length of each validation block
N_FOLDS = 3
CALIBRATION_DAYS = 365  # window used to pick and label SKUs; ends before fold 1 starts
INTERMITTENT_ZERO_SHARE = 0.5

# Reorder simulation (all simulated; see docs/assumptions.md)
LEAD_TIME = 2           # days from order to availability
START_COVER_DAYS = 3    # starting stock = this many days of trailing 28-day mean sales
SWEEP = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
