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
