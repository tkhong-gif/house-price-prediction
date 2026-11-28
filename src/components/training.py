"""Stage 4: train random forest and gradient boosting models for `total_value` on two feature sets, with and without the neighborhood price.
The data is split 80/20 at random, features from sale prices are recomputed from
training rows only, and settings are tuned with RandomizedSearchCV. Models go to
output/models/ and scores to reports/training_metrics.csv."""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.feature_engineering import add_neighborhood_aggregates, add_price_momentum
from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

TARGET = "total_value"
NEIGHBORHOOD_PRICE = "neighborhood_median_price_per_sqft"
SALE_DERIVED_FEATURES = [NEIGHBORHOOD_PRICE, "neighborhood_sale_count", "price_momentum"]

CATEGORICAL_FEATURES = ["land_use_class", "area"]
NUMERIC_FEATURES = [
    "acreage", "year_built", "heated_area", "bedrooms", "bathrooms",
    "latitude", "longitude", "property_age",
    "neighborhood_median_price_per_sqft", "neighborhood_sale_count", "price_momentum",
    "education_nearest_miles", "places_count", "health_care_share", "activity_share",
    "emergency_fire_miles", "emergency_police_miles", "emergency_medical_miles",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FEATURE_SETS = {
    "with_neighborhood_price": FEATURE_COLUMNS,
    "without_neighborhood_price": [c for c in FEATURE_COLUMNS if c != NEIGHBORHOOD_PRICE],
}
MODEL_TYPES = ["random_forest", "gradient_boosting"]


def load_training_data(features_path, filename="properties_features.csv"):
    """Rows with a positive target from a feature table (the full one by default, or
    e.g. data/sample/properties_sample.csv). The file's sale-derived features are
    dropped - they were computed on every row, so split_with_aggregates recomputes them."""
    df = read_dataframe(features_path, filename)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    return df.drop(columns=SALE_DERIVED_FEATURES)


def split_with_aggregates(df, training_cfg):
    """Split rows into train and test (config test_size and random_state), then add
    the neighborhood price features and price_momentum to both, computed from the
    training rows only."""
    train, test = train_test_split(
        df, test_size=training_cfg["test_size"], random_state=training_cfg["random_state"]
    )

    def add(rows):
        """Add the sale-derived features to `rows`, using the training rows as the source."""
        return add_price_momentum(add_neighborhood_aggregates(rows, source=train), source=train)

    return add(train), add(test)


def build_preprocessor(features):
    """Preprocessing for the given feature columns: CATEGORICAL_FEATURES get
    most-frequent imputation and one-hot encoding (categories not seen in training are
    ignored), every other column median imputation and standard scaling."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, [c for c in features if c not in CATEGORICAL_FEATURES]),
        ("cat", categorical_pipeline, [c for c in features if c in CATEGORICAL_FEATURES]),
    ])


def eval_metrics(y_true, y_pred):
    """RMSE, MAE, R squared and mean absolute percentage error of predictions against
    the actual values."""
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}
GRADIENT_BOOSTING_BASE = dict(max_iter=3000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)
GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [31, 63, 127],
    "min_samples_leaf": [10, 20, 50],
    "l2_regularization": [0.0, 1.0, 5.0],
    "max_features": [0.5, 0.8, 1.0],
}
XGBOOST_PARAM_DISTRIBUTIONS = {
    "n_estimators": [400, 700, 1000],
    "max_depth": [5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5, 10],
}

DEFAULT_PARAMS = {
    "random_forest": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.5},
    "gradient_boosting": {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 10,
                          "l2_regularization": 1.0, "max_features": 0.5},
}


def make_model(model_name, params=None, random_state=0):
    """An unfitted model of type `model_name` with fixed settings (default:
    DEFAULT_PARAMS), using every CPU core."""
    if model_name not in DEFAULT_PARAMS:
        raise ValueError(f"no fixed-settings model for {model_name!r}, expected one of {list(DEFAULT_PARAMS)}")
    params = DEFAULT_PARAMS[model_name] if params is None else params
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    if model_name == "gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state, **GRADIENT_BOOSTING_BASE, **params)
    raise AssertionError("unreachable")


def build_search_estimators(random_state, training_cfg, model_types=None):
    """A hyperparameter search (RandomizedSearchCV) per model type in `model_types`
    (default: MODEL_TYPES), each trying config `search_iter` settings from its
    parameter grid with `cv_folds`-fold cross-validation, scored by RMSE. xgboost is
    only imported when it is asked for, so it doesn't have to be installed otherwise."""
    model_types = MODEL_TYPES if model_types is None else model_types
    search_kwargs = dict(
        n_iter=training_cfg["search_iter"], cv=training_cfg["cv_folds"],
        scoring="neg_root_mean_squared_error", random_state=random_state, n_jobs=-1, refit=True,
    )
    searches = {}
    if "random_forest" in model_types:
        searches["random_forest"] = RandomizedSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            RANDOM_FOREST_PARAM_DISTRIBUTIONS, **search_kwargs,
        )
    if "gradient_boosting" in model_types:
        searches["gradient_boosting"] = RandomizedSearchCV(
            HistGradientBoostingRegressor(random_state=random_state, **GRADIENT_BOOSTING_BASE),
            GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS, **search_kwargs,
        )
    if "xgboost" in model_types:
        from xgboost import XGBRegressor

        searches["xgboost"] = RandomizedSearchCV(
            XGBRegressor(random_state=random_state, n_jobs=1, tree_method="approx"),
            XGBOOST_PARAM_DISTRIBUTIONS, **search_kwargs,
        )
    return searches


def train_models(train, test, training_cfg):
    """For each feature set: fit its preprocessor, tune and fit every model in
    MODEL_TYPES on log(1 + total_value), and score it on the test set in dollars.

    Returns (metrics, models, preprocessors, best hyperparameters). Metrics and
    models are keyed by (feature_set, model_name), preprocessors by feature set."""
    results = {}
    models = {}
    preprocessors = {}
    best_params = {}
    y_train_log = np.log1p(train[TARGET])

    for set_name, features in FEATURE_SETS.items():
        preprocessor = build_preprocessor(features)
        X_train_p = preprocessor.fit_transform(train[features])
        X_test_p = preprocessor.transform(test[features])

        searches = build_search_estimators(training_cfg["random_state"], training_cfg)
        for model_name in MODEL_TYPES:
            search = searches[model_name]
            search.fit(X_train_p, y_train_log)
            best_model = search.best_estimator_
            preds = np.expm1(best_model.predict(X_test_p))
            results[(set_name, model_name)] = eval_metrics(test[TARGET], preds)
            models[(set_name, model_name)] = best_model
            best_params[f"{set_name}__{model_name}"] = search.best_params_

        preprocessors[set_name] = preprocessor

    return results, models, preprocessors, best_params


def save_artifacts(models_path, preprocessors, models):
    """Save each preprocessor and model to `models_path` as
    {feature_set}_preprocessor.joblib and {feature_set}_{model_name}.joblib."""
    os.makedirs(models_path, exist_ok=True)
    for set_name, preprocessor in preprocessors.items():
        joblib.dump(preprocessor, os.path.join(models_path, f"{set_name}_preprocessor.joblib"))
    for (set_name, model_name), model in models.items():
        joblib.dump(model, os.path.join(models_path, f"{set_name}_{model_name}.joblib"))


def run():
    """Stage 4: load the feature table, split it, tune and train every model, print
    the metrics, and save the models, reports/training_metrics.csv and
    reports/best_hyperparameters.json."""
    cfg = load_config()
    training_cfg = cfg["training"]
    features_path = resolve_path(cfg, "features_data")
    models_path = resolve_path(cfg, "models")
    reports_path = resolve_path(cfg, "reports")

    train, test = split_with_aggregates(load_training_data(features_path), training_cfg)
    results, models, preprocessors, best_params = train_models(train, test, training_cfg)

    metrics_df = pd.DataFrame(results).T
    metrics_df.index = metrics_df.index.set_names(["feature_set", "model"])
    metrics_df = metrics_df.sort_values(["feature_set", "rmse"])
    print(metrics_df)
    print()
    print(json.dumps(best_params, indent=2))

    save_artifacts(models_path, preprocessors, models)
    save_dataframe(metrics_df.reset_index(), reports_path, "training_metrics.csv")
    os.makedirs(reports_path, exist_ok=True)
    with open(os.path.join(reports_path, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    return metrics_df


if __name__ == "__main__":
    run()
