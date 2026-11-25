"""Tests for training, prediction, analysis and the pipeline runner, on the sample data."""

import os

import numpy as np
import pandas as pd
import pytest

from src.components import analysis, predict
from src.components.training import (
    FEATURE_SETS, MODEL_TYPES, NEIGHBORHOOD_PRICE, TARGET, build_preprocessor, build_search_estimators,
    eval_metrics, load_training_data, make_model, save_artifacts, split_with_aggregates,
)
from src.pipeline.run_pipeline import DEFAULT_STAGES, select_stages
from src.utils.load_config import load_config

SAMPLE_DIR = os.path.join(load_config()["paths"]["root"], "data", "sample")
SMALL_FOREST = {"n_estimators": 10, "max_depth": 8, "min_samples_leaf": 2, "max_features": 0.5}
SMALL_BOOSTING = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                  "l2_regularization": 1.0, "max_features": 0.5}


@pytest.fixture(scope="module")
def sample():
    return load_training_data(SAMPLE_DIR, "properties_sample.csv")


@pytest.fixture(scope="module")
def splits(sample):
    return split_with_aggregates(sample, load_config()["training"])


def test_sample_has_every_feature(sample):
    assert len(sample) > 9_000
    assert (sample[TARGET] > 0).all()
    missing = set(FEATURE_SETS["with_neighborhood_price"]) - set(sample.columns) - {
        NEIGHBORHOOD_PRICE, "neighborhood_sale_count", "price_momentum"}
    assert not missing


def test_split_has_no_overlap_and_recomputes_aggregates(splits):
    train, test = splits
    assert len(train) + len(test) > 9_000
    assert NEIGHBORHOOD_PRICE in test and "price_momentum" in test
    # a neighborhood whose only sales are in the test split gets no neighborhood price
    train_nbhds = set(train.loc[train["price_per_sqft"].notna(), "neighborhood_code"])
    unseen = test[~test["neighborhood_code"].isin(train_nbhds)]
    assert unseen[NEIGHBORHOOD_PRICE].isna().all()


def test_preprocessor_handles_missing_values_and_unknown_area(splits):
    train, _ = splits
    features = FEATURE_SETS["with_neighborhood_price"]
    pre = build_preprocessor(features).fit(train[features])
    row = pd.DataFrame([{c: np.nan for c in features} | {"area": "NOT_AN_AREA"}])
    assert np.isfinite(pre.transform(row)).all()


def test_eval_metrics_perfect_prediction():
    m = eval_metrics(np.array([100.0, 200.0]), np.array([100.0, 200.0]))
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1 and m["mape"] == 0


def test_model_beats_baseline_and_predict_round_trip(splits, tmp_path, monkeypatch):
    train, test = splits
    features = FEATURE_SETS["with_neighborhood_price"]
    preds, pre, model = analysis.fit_predict(train, test, features, "random_forest", SMALL_FOREST)
    baseline = analysis.baseline_predict(train, test)
    assert eval_metrics(test[TARGET], preds)["mae"] < eval_metrics(test[TARGET], baseline)["mae"]

    # save like quick_train.py does, then load through predict.py from a temporary folder
    save_artifacts(str(tmp_path) + "/demo/", {"with_neighborhood_price": pre}, {("with_neighborhood_price", "random_forest"): model})
    monkeypatch.setattr(predict, "resolve_path", lambda cfg, key: str(tmp_path) + "/")
    predict.load_artifact.cache_clear()
    value = predict.predict_total_value(predict.EXAMPLE_HOUSE, models_dir="demo/")
    predict.load_artifact.cache_clear()
    assert 50_000 < value < 2_000_000

    with pytest.raises(ValueError):
        predict.load_model("not_a_feature_set")


def test_best_model_name_prefers_trained_gradient_boosting(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "resolve_path", lambda cfg, key: str(tmp_path) + "/")
    assert predict.best_model_name() == "gradient_boosting"  # nothing trained: first preference
    (tmp_path / "with_neighborhood_price_random_forest.joblib").write_bytes(b"")
    assert predict.best_model_name() == "random_forest"  # the only one trained
    (tmp_path / "with_neighborhood_price_gradient_boosting.joblib").write_bytes(b"")
    pd.DataFrame({"feature_set": ["with_neighborhood_price"] * 2, "model": ["random_forest", "gradient_boosting"],
                  "rmse": [1.0, 2.0]}).to_csv(tmp_path / "training_metrics.csv", index=False)
    assert predict.best_model_name() == "random_forest"  # the training report decides


def test_analysis_tables(splits):
    train, test = splits
    features = FEATURE_SETS[analysis.GROUP_FEATURE_SET]
    preds, pre, model = analysis.fit_predict(train, test, features, "gradient_boosting", SMALL_BOOSTING)
    table = analysis.error_table(test.assign(estimate=preds), "estimate", "area")
    assert table["houses"].sum() == len(test)
    assert table["within_10pct"].between(0, 1).all()

    same_house = analysis.same_house_by_area(train, pre, model)
    assert set(same_house["area"]) == set(train["area"])
    assert (same_house["estimate"] > 0).all()

    bands = analysis.price_band(test[TARGET])
    assert bands.nunique() == analysis.PRICE_BANDS


def test_feature_groups_cover_the_group_feature_set():
    grouped = [c for cols in analysis.FEATURE_GROUPS.values() for c in cols]
    assert sorted(grouped) == sorted(FEATURE_SETS[analysis.GROUP_FEATURE_SET])


def test_select_stages():
    assert select_stages() == DEFAULT_STAGES
    assert select_stages(start="train") == ["train", "evaluate", "analyze"]
    assert select_stages(only="finetune") == ["finetune"]
    with pytest.raises(SystemExit):
        select_stages(only="bogus")


def test_every_model_type_trains_and_boosting_beats_forest(splits):
    train, test = splits
    features = FEATURE_SETS["with_neighborhood_price"]
    mae = {}
    for name in MODEL_TYPES:
        preds = analysis.fit_predict(train, test, features, name)[0]
        mae[name] = eval_metrics(test[TARGET], preds)["mae"]
    assert mae["gradient_boosting"] < mae["random_forest"]


def test_search_estimators_cover_model_types():
    searches = build_search_estimators(0, load_config()["training"])
    assert set(searches) == set(MODEL_TYPES)
    with pytest.raises(ValueError):
        make_model("not_a_model")
