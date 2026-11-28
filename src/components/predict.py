"""Estimate `total_value` for one house with a saved preprocessor and model from output/models/.
Missing features are filled as in training, but estimates are much weaker without the
location fields."""

import functools
import os

import joblib
import numpy as np
import pandas as pd

from src.components.training import FEATURE_SETS, MODEL_TYPES
from src.utils.common import resolve_path
from src.utils.load_config import load_config

DEFAULT_FEATURE_SET = "with_neighborhood_price"
MODEL_PREFERENCE = ["gradient_boosting", "random_forest"]

EXAMPLE_HOUSE = {
    "land_use_class": "single_family",
    "area": "CHARLOTTE_1",
    "acreage": 0.46,
    "year_built": 1950,
    "heated_area": 1017,
    "bedrooms": 2,
    "bathrooms": 1,
    "latitude": 35.3339,
    "longitude": -80.8713,
    "property_age": 76,
    "neighborhood_median_price_per_sqft": 161.3,
    "neighborhood_sale_count": 46,
    "price_momentum": 0.2,
    "education_nearest_miles": 0.5,
    "places_count": 6000,
    "health_care_share": 0.06,
    "activity_share": 0.03,
    "emergency_fire_miles": 1.2,
    "emergency_police_miles": 2.0,
    "emergency_medical_miles": 2.5,
}


@functools.lru_cache(maxsize=None)
def load_artifact(models_path, filename):
    """Cached joblib load, so repeated calls reuse the same in-memory
    preprocessor/model instead of reloading it from disk."""
    return joblib.load(models_path + filename)


def best_model_name(feature_set=DEFAULT_FEATURE_SET, models_dir=""):
    """The model type to use when none is named: the one with the lowest test RMSE for
    this feature set in reports/training_metrics.csv (main models only), else the first
    of MODEL_PREFERENCE whose file exists, else the first of MODEL_PREFERENCE."""
    cfg = load_config()
    models_path = resolve_path(cfg, "models") + models_dir
    trained = [m for m in MODEL_TYPES if os.path.exists(f"{models_path}{feature_set}_{m}.joblib")]
    metrics_file = os.path.join(resolve_path(cfg, "reports"), "training_metrics.csv")
    if not models_dir and os.path.exists(metrics_file):
        metrics = pd.read_csv(metrics_file)
        metrics = metrics[(metrics["feature_set"] == feature_set) & metrics["model"].isin(trained)]
        if len(metrics):
            return metrics.loc[metrics["rmse"].idxmin(), "model"]
    return next((m for m in MODEL_PREFERENCE if m in trained), MODEL_PREFERENCE[0])


def load_model(feature_set=DEFAULT_FEATURE_SET, model_name=None, models_dir=""):
    """(preprocessor, model) for a feature set, from output/models/ (or its
    `models_dir` subfolder, e.g. "demo/" for bin/quick_train.py's models).
    model_name None picks the best trained model (see best_model_name)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}, expected one of {list(FEATURE_SETS)}")
    if model_name is None:
        model_name = best_model_name(feature_set, models_dir)
    if model_name not in MODEL_TYPES:
        raise ValueError(f"model {model_name!r} is not in use, expected one of {MODEL_TYPES}")
    models_path = resolve_path(load_config(), "models") + models_dir
    preprocessor = load_artifact(models_path, f"{feature_set}_preprocessor.joblib")
    model = load_artifact(models_path, f"{feature_set}_{model_name}.joblib")
    return preprocessor, model


def predict_total_value(features, feature_set=DEFAULT_FEATURE_SET, model_name=None, models_dir=""):
    """Estimate one house's value in dollars from a dict of feature values (missing keys become NaN).
    model_name None uses the best trained model, and models_dir picks a subfolder of
    output/models/ ("demo/" for the demo)."""
    preprocessor, model = load_model(feature_set, model_name, models_dir)
    row = pd.DataFrame([{col: features.get(col, np.nan) for col in FEATURE_SETS[feature_set]}])
    log_pred = model.predict(preprocessor.transform(row))[0]
    return float(np.expm1(log_pred))


if __name__ == "__main__":
    for feature_set in FEATURE_SETS:
        for model_name in MODEL_TYPES:
            try:
                pred = predict_total_value(EXAMPLE_HOUSE, feature_set=feature_set, model_name=model_name)
            except FileNotFoundError:
                print(f"{feature_set} {model_name}: not trained yet")
                continue
            print(f"{feature_set} {model_name}: ${pred:,.0f}")
