"""Get-started demo: train both model types on the 10,000-house sample and print their test accuracy.
It runs the same steps as training.py with fixed settings instead of a search, in under
a minute, and saves the models to output/models/demo/."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.components.training import (
    FEATURE_SETS, MODEL_TYPES, TARGET,
    build_preprocessor, eval_metrics, load_training_data, make_model, save_artifacts, split_with_aggregates,
)
from src.utils.common import resolve_path
from src.utils.load_config import load_config

FEATURE_SET = "with_neighborhood_price"
DEMO_DIR = "demo/"


def main():
    """Train the demo models on the sample, print their test metrics and save them."""
    cfg = load_config()
    df = load_training_data(os.path.join(cfg["paths"]["root"], "data", "sample"), "properties_sample.csv")
    print(f"loaded {len(df):,} houses from the sample")

    train, test = split_with_aggregates(df, cfg["training"])
    features = FEATURE_SETS[FEATURE_SET]
    print(f"trained on {len(train):,} houses, tested on {len(test):,}")

    preprocessor = build_preprocessor(features)
    X_train = preprocessor.fit_transform(train[features])
    X_test = preprocessor.transform(test[features])

    models = {}
    print(f"  {'':16} {'RMSE':>10} {'MAE':>10} {'R squared':>10} {'avg error':>10}")
    for model_name in MODEL_TYPES:
        model = make_model(model_name, random_state=cfg["training"]["random_state"])
        model.fit(X_train, np.log1p(train[TARGET]))
        m = eval_metrics(test[TARGET], np.expm1(model.predict(X_test)))
        print(f"  {model_name:16} {m['rmse']:>10,.0f} {m['mae']:>10,.0f} {m['r2']:>10.3f} {m['mape']:>10.1%}")
        models[(FEATURE_SET, model_name)] = model

    models_path = resolve_path(cfg, "models") + DEMO_DIR
    save_artifacts(models_path, {FEATURE_SET: preprocessor}, models)
    print(f"saved demo models to {models_path}")


if __name__ == "__main__":
    main()
