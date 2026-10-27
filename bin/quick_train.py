"""Get-started demo: train a small random forest on the 10,000-house sample in
data/sample/ and print its test-set accuracy. Takes well under a minute.

It runs the same steps as src/components/training.py on less data and with fixed
settings instead of a hyperparameter search:

1. Split the sample 80/20 into train and test houses.
2. Recompute the neighborhood price features from training houses only.
3. Fit the preprocessor (impute missing values, scale numbers, one-hot encode
   categories) and a random forest on log(total_value).
4. Report RMSE, MAE, R squared and average percent error on the test houses.
5. Save the model to output/models/demo/, where bin/predict_example.py --demo uses it.

Usage (from the repository root):
    uv run python bin/quick_train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.components.training import (
    FEATURE_SETS, SALE_DERIVED_FEATURES, TARGET,
    build_preprocessor, eval_metrics, save_artifacts, split_with_aggregates,
)
from sklearn.ensemble import RandomForestRegressor
from src.utils.common import read_dataframe, resolve_path
from src.utils.load_config import load_config

FEATURE_SET = "with_neighborhood_price"
# Small fixed settings so the demo is fast. The full run in training.py searches
# these instead (see RANDOM_FOREST_PARAM_DISTRIBUTIONS).
DEMO_PARAMS = {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 2, "max_features": 0.5}
DEMO_DIR = "demo/"


def main():
    """Train the demo model on the sample, print its test metrics and save it."""
    cfg = load_config()
    df = read_dataframe(os.path.join(cfg["paths"]["root"], "data", "sample"), "properties_sample.csv")
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].drop(columns=SALE_DERIVED_FEATURES)
    print(f"loaded {len(df):,} houses from the sample")

    train, test = split_with_aggregates(df, cfg["training"])
    features = FEATURE_SETS[FEATURE_SET]

    preprocessor = build_preprocessor(features)
    X_train = preprocessor.fit_transform(train[features])
    X_test = preprocessor.transform(test[features])

    model = RandomForestRegressor(random_state=cfg["training"]["random_state"], n_jobs=-1, **DEMO_PARAMS)
    model.fit(X_train, np.log1p(train[TARGET]))
    preds = np.expm1(model.predict(X_test))

    metrics = eval_metrics(test[TARGET], preds)
    print(f"trained on {len(train):,} houses, tested on {len(test):,}")
    print(f"  RMSE             ${metrics['rmse']:,.0f}")
    print(f"  MAE              ${metrics['mae']:,.0f}")
    print(f"  R squared        {metrics['r2']:.3f}")
    print(f"  average error    {metrics['mape']:.1%}")

    models_path = resolve_path(cfg, "models") + DEMO_DIR
    save_artifacts(models_path, {FEATURE_SET: preprocessor}, {(FEATURE_SET, "random_forest"): model})
    print(f"saved demo model to {models_path}")


if __name__ == "__main__":
    main()
