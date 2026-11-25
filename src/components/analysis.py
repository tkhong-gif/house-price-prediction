"""Stage 6: how reliable the model is, and what each group of features is worth.

training.py reports one test score from one split. This stage answers the questions
that one number can't:

1. Is the model better than a simple rule? A baseline (heated area x the
   neighborhood's median price per sq ft, scaled to assessed values) is scored on the
   same splits as the models.
2. How stable is the score? Everything is repeated over `n_splits` random 80/20
   splits; split 0 is the same split training.py uses.
3. Where is the model weak? Errors on split 0 by area, price band and land use.
4. What is each feature group worth? (the research question)
   - drop-column test: retrain without the group and compare test MAE on the same
     splits
   - group permutation importance next to a pure-noise column
   The rule for "this group matters" is fixed before looking at results
   (MATTERS_MIN_MAE_INCREASE): removing it raises test MAE by at least 1% in most
   splits, and shuffling it hurts more than shuffling the noise column.
5. Same house, different areas: one typical house, estimated with each area's
   typical location and accessibility values.

Every model type in training.MODEL_TYPES is scored on both feature sets. The error
breakdowns use BEST_MODEL, and the feature-group questions use GROUP_MODEL on the
without_neighborhood_price feature set, because the neighborhood price already
contains most of the effect of location and amenities (see training.py).

Models here use fixed settings (training.DEFAULT_PARAMS) instead of a hyperparameter
search, so many of them can be trained in reasonable time. Their scores are slightly
below the tuned models in output/models/.

Usage (from the repository root):
    uv run python -m src.components.analysis --sample    # 10,000-house sample, a few minutes
    uv run python -m src.components.analysis             # full feature table, much longer
    uv run python -m src.components.analysis --splits 3 --skip-drop-column

Outputs go to reports/analysis/ (reports/analysis_sample/ with --sample):
summary.csv, split_metrics.csv, errors_by_*.csv, drop_column.csv,
group_importance.csv, same_house_by_area.csv, and report.md, which puts them together.
"""

import argparse
import os

import numpy as np
import pandas as pd
from src.components.training import (
    DEFAULT_PARAMS, FEATURE_SETS, MODEL_TYPES, NEIGHBORHOOD_PRICE, TARGET,
    build_preprocessor, eval_metrics, load_training_data, make_model, split_with_aggregates,
)
from src.utils.common import resolve_path, save_dataframe
from src.utils.load_config import load_config

N_SPLITS = 5

# The most accurate model on the sample; its errors are broken down by area and price.
BEST_MODEL = "with_neighborhood_price__gradient_boosting"
# The model type used for the feature-group tests and the same-house comparison.
GROUP_MODEL = "gradient_boosting"
GROUP_FEATURE_SET = "without_neighborhood_price"
FEATURE_GROUPS = {
    "house": ["land_use_class", "acreage", "year_built", "heated_area", "bedrooms", "bathrooms", "property_age"],
    "location": ["area", "latitude", "longitude"],
    "neighborhood_trend": ["neighborhood_sale_count", "price_momentum"],
    "education": ["education_nearest_miles"],
    "health_care": ["health_care_share"],
    "emergency": ["emergency_fire_miles", "emergency_police_miles", "emergency_medical_miles"],
    "activity": ["activity_share"],
    "density": ["places_count"],
}
NOISE = "noise"

# Decision rule, fixed in advance: a group matters if dropping it raises test MAE by at
# least this fraction in more than half of the splits, and it ranks above the noise column.
MATTERS_MIN_MAE_INCREASE = 0.01
PERMUTATION_ROWS = 5_000
PERMUTATION_REPEATS = 3

PRICE_BANDS = 5
WITHIN = 0.10  # share of houses estimated within 10% of their value

# Columns that describe where a house is; the same-house comparison sets these to each
# area's typical (median) values and keeps the house itself fixed.
LOCATION_COLUMNS = [
    "latitude", "longitude", "neighborhood_sale_count", "price_momentum",
    "education_nearest_miles", "places_count", "health_care_share", "activity_share",
    "emergency_fire_miles", "emergency_police_miles", "emergency_medical_miles",
]
HOUSE_COLUMNS = ["acreage", "year_built", "heated_area", "bedrooms", "bathrooms", "property_age"]


def load_data(sample=False):
    """(rows, report folder): the full feature table, or the 10,000-house sample."""
    cfg = load_config()
    if sample:
        path = os.path.join(cfg["paths"]["root"], "data", "sample")
        df = load_training_data(path, "properties_sample.csv")
    else:
        df = load_training_data(resolve_path(cfg, "features_data"))
    reports = os.path.join(resolve_path(cfg, "reports"), "analysis_sample" if sample else "analysis")
    return df, reports


def fit_predict(train, test, features, model_name=GROUP_MODEL, params=None, random_state=0):
    """Fit a preprocessor and a `model_name` model (settings: `params`, default
    training.DEFAULT_PARAMS) on log(1 + value) and return (dollar predictions for
    `test`, preprocessor, model)."""
    preprocessor = build_preprocessor(features)
    model = make_model(model_name, params, random_state)
    model.fit(preprocessor.fit_transform(train[features]), np.log1p(train[TARGET]))
    preds = np.expm1(model.predict(preprocessor.transform(test[features])))
    return preds, preprocessor, model


def baseline_predict(train, test):
    """heated_area x neighborhood median price per sq ft x one scale factor.

    The neighborhood median comes from sale prices, which are mostly older and lower
    than the 2023 assessment, so the product is scaled by the median ratio of value to
    product on the training rows. A missing neighborhood price uses the area's median,
    then the county's; a missing heated area the training median."""
    area_ppsf = train.groupby("area")[NEIGHBORHOOD_PRICE].median()

    def product(rows):
        """heated_area x neighborhood price per sq ft, with fallbacks for missing values."""
        ppsf = rows[NEIGHBORHOOD_PRICE].fillna(rows["area"].map(area_ppsf)).fillna(train[NEIGHBORHOOD_PRICE].median())
        return rows["heated_area"].fillna(train["heated_area"].median()) * ppsf

    scale = (train[TARGET] / product(train)).median()
    return (product(test) * scale).to_numpy()


def split(df, training_cfg, i):
    """Split i of the repeated splits; split 0 is the one training.py uses."""
    return split_with_aggregates(df, {**training_cfg, "random_state": training_cfg["random_state"] + i})


def repeated_splits(df, training_cfg, n_splits=N_SPLITS, drop_column=True):
    """Score the baseline, every feature set x model type (named
    "<feature_set>__<model>") and (if drop_column) GROUP_MODEL on the group feature set
    without each FEATURE_GROUPS group, on n_splits splits.

    Returns (metrics per split and model, split 0's test rows with each model's
    estimate, split 0's train rows, split 0's models keyed by name)."""
    rows = []
    first = None
    for i in range(n_splits):
        train, test = split(df, training_cfg, i)
        seed = training_cfg["random_state"] + i
        estimates = {"baseline": baseline_predict(train, test)}
        models = {}
        for set_name, features in FEATURE_SETS.items():
            for model_name in MODEL_TYPES:
                name = f"{set_name}__{model_name}"
                preds, preprocessor, model = fit_predict(train, test, features, model_name, random_state=seed)
                estimates[name], models[name] = preds, (preprocessor, model)
        if drop_column:
            for group, columns in FEATURE_GROUPS.items():
                features = [c for c in FEATURE_SETS[GROUP_FEATURE_SET] if c not in columns]
                estimates[f"drop:{group}"] = fit_predict(train, test, features, GROUP_MODEL, random_state=seed)[0]
        for name, preds in estimates.items():
            rows.append({"split": i, "model": name, **eval_metrics(test[TARGET], preds)})
        print(f"split {i + 1}/{n_splits} done")
        if i == 0:
            first = (test.assign(**{f"estimate:{k}": v for k, v in estimates.items()}), train, models)
    return (pd.DataFrame(rows), *first)


def summarize_splits(metrics):
    """Mean and standard deviation of each metric per model, over the splits."""
    summary = metrics.drop(columns="split").groupby("model").agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    models = [f"{s}__{m}" for s in FEATURE_SETS for m in MODEL_TYPES]
    order = ["baseline", *models] + [m for m in summary.index if m.startswith("drop:")]
    return summary.reindex([m for m in order if m in summary.index]).reset_index()


def error_table(test, estimate_col, by):
    """Error of one model's estimates on `test`, grouped by column `by`."""
    pct = (test[estimate_col] - test[TARGET]).abs() / test[TARGET]
    frame = pd.DataFrame({by: test[by], "abs_error": (test[estimate_col] - test[TARGET]).abs(), "pct": pct})
    return frame.groupby(by, observed=True).agg(
        houses=("pct", "size"),
        mae=("abs_error", "mean"),
        mape=("pct", "mean"),
        median_pct_error=("pct", "median"),
        within_10pct=("pct", lambda p: (p <= WITHIN).mean()),
    ).reset_index()


def price_band(values, n=PRICE_BANDS):
    """Label each value with its quantile band, e.g. '$250k-$330k'."""
    bands = pd.qcut(values, n)
    labels = {b: f"${b.left / 1000:,.0f}k-${b.right / 1000:,.0f}k" for b in bands.cat.categories}
    return bands.map(labels)


def drop_column_results(metrics):
    """Per group: MAE increase when it is dropped, relative to the full group feature
    set on the same split, and in how many splits it reached MATTERS_MIN_MAE_INCREASE."""
    mae = metrics.pivot(index="split", columns="model", values="mae")
    full = mae[f"{GROUP_FEATURE_SET}__{GROUP_MODEL}"]
    rows = []
    for group in FEATURE_GROUPS:
        increase = mae[f"drop:{group}"] / full - 1
        rows.append({
            "group": group,
            "features": ", ".join(FEATURE_GROUPS[group]),
            "mae_increase_mean": increase.mean(),
            "mae_increase_min": increase.min(),
            "mae_increase_max": increase.max(),
            "splits_over_threshold": int((increase >= MATTERS_MIN_MAE_INCREASE).sum()),
            "splits": len(increase),
        })
    return pd.DataFrame(rows)


def group_importance(train, test, training_cfg):
    """Permutation importance of each feature group and of a pure-noise column.

    A model is trained on the group feature set plus NOISE (random numbers). Each group's
    columns are then shuffled together on up to PERMUTATION_ROWS test houses, and the
    rise in MAE is its importance. A group that doesn't beat the noise column carries no
    more information than random numbers."""
    rng = np.random.default_rng(training_cfg["random_state"])
    train = train.assign(**{NOISE: rng.normal(size=len(train))})
    test = test.assign(**{NOISE: rng.normal(size=len(test))})
    test = test.sample(min(PERMUTATION_ROWS, len(test)), random_state=training_cfg["random_state"])
    features = FEATURE_SETS[GROUP_FEATURE_SET] + [NOISE]
    preds, preprocessor, model = fit_predict(train, test, features, GROUP_MODEL, random_state=training_cfg["random_state"])
    base_mae = np.mean(np.abs(preds - test[TARGET]))

    rows = []
    for group, columns in {**FEATURE_GROUPS, NOISE: [NOISE]}.items():
        increases = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = test.copy()
            order = rng.permutation(len(test))
            shuffled[columns] = test[columns].to_numpy()[order]
            mae = np.mean(np.abs(np.expm1(model.predict(preprocessor.transform(shuffled[features]))) - test[TARGET]))
            increases.append(mae / base_mae - 1)
        rows.append({"group": group, "mae_increase_when_shuffled": np.mean(increases)})
    return pd.DataFrame(rows).sort_values("mae_increase_when_shuffled", ascending=False).reset_index(drop=True)


def verdicts(drop, importance):
    """Apply the decision rule: matters = dropping it raises MAE by at least
    MATTERS_MIN_MAE_INCREASE in more than half of the splits AND it beats the noise column."""
    noise = importance.loc[importance["group"] == NOISE, "mae_increase_when_shuffled"].iloc[0]
    out = drop.merge(importance, on="group", how="left")
    out["beats_noise"] = out["mae_increase_when_shuffled"] > noise
    out["matters"] = (out["splits_over_threshold"] > out["splits"] / 2) & out["beats_noise"]
    return out


def same_house_by_area(train, preprocessor, model):
    """One typical single-family house (training medians), estimated with each area's
    median location and accessibility values, by the group feature set's model."""
    houses = train[train["land_use_class"] == "single_family"]
    house = houses[HOUSE_COLUMNS].median()
    by_area = train.groupby("area")[LOCATION_COLUMNS].median()
    rows = pd.DataFrame([{**house, **loc, "area": area, "land_use_class": "single_family"} for area, loc in by_area.iterrows()])
    features = FEATURE_SETS[GROUP_FEATURE_SET]
    rows["estimate"] = np.expm1(model.predict(preprocessor.transform(rows[features])))
    rows["vs_median_area"] = rows["estimate"] / rows["estimate"].median() - 1
    described = f"{house['heated_area']:,.0f} sq ft, {house['bedrooms']:.0f} bed, {house['bathrooms']:g} bath, built {house['year_built']:.0f}"
    print(f"same-house comparison uses a typical single-family house: {described}")
    return rows[["area", "estimate", "vs_median_area"]].sort_values("estimate", ascending=False).reset_index(drop=True)


def pct(x):
    """Format a fraction as a percentage."""
    return f"{x:.1%}"


def money(x):
    """Format a number as whole dollars."""
    return f"${x:,.0f}"


def markdown_table(df, formats):
    """A DataFrame as a markdown table, formatting the columns named in `formats`."""
    shown = df.copy()
    for col, fmt in formats.items():
        if col in shown:
            shown[col] = shown[col].map(fmt)
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "|" + "---|" * len(shown.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def write_report(path, n_rows, n_splits, summary, by_area, by_price, by_use, decided, same_house):
    """Write report.md: every result table, with a one-line reading of each."""
    metric_formats = {c: (pct if "mape" in c else (lambda x: f"{x:.3f}") if "r2" in c else money)
                      for c in summary.columns if c != "model"}
    error_formats = {"mae": money, "mape": pct, "median_pct_error": pct, "within_10pct": pct}
    best = summary.set_index("model")
    gain = 1 - best.loc[BEST_MODEL, "mae_mean"] / best.loc["baseline", "mae_mean"]
    settings = "; ".join(f"{m}: {DEFAULT_PARAMS[m]}" for m in MODEL_TYPES)

    parts = [
        "# Model analysis",
        f"{n_rows:,} houses, {n_splits} random 80/20 splits, models with fixed settings "
        f"({settings}). Generated by `src/components/analysis.py`.",
        "## 1. Models against a simple baseline (mean and standard deviation over the splits)",
        "`baseline` is heated area x neighborhood median price per sq ft, scaled to assessed values. "
        "Model names are `<feature set>__<model type>`.",
        markdown_table(summary, metric_formats),
        f"The {BEST_MODEL} model's MAE is {pct(gain)} lower than the baseline's.",
    ]
    if by_area is not None:
        parts += [
            f"## 2. Where the {BEST_MODEL} model is weak (split 0 test houses)",
            "### By area", markdown_table(by_area, error_formats),
            "### By price band (actual assessed value)", markdown_table(by_price, error_formats),
            "### By land use", markdown_table(by_use, error_formats),
        ]
    if decided is not None:
        parts += [
            f"## 3. What each feature group is worth ({GROUP_MODEL}, {GROUP_FEATURE_SET} feature set)",
            f"A group **matters** if dropping it raises test MAE by at least "
            f"{pct(MATTERS_MIN_MAE_INCREASE)} in more than half of the splits, and shuffling it "
            "raises MAE more than shuffling a column of random numbers.",
            markdown_table(decided, {"mae_increase_mean": pct, "mae_increase_min": pct, "mae_increase_max": pct,
                                     "mae_increase_when_shuffled": pct}),
            "Groups whose information overlaps (for example location and the accessibility "
            "features) can each look unimportant when dropped alone, because the others make up for it.",
        ]
    if same_house is not None:
        parts += [
            "## 4. The same house in each area",
            "One typical single-family house, estimated with each area's median location and "
            f"accessibility values ({GROUP_FEATURE_SET}__{GROUP_MODEL} model, split 0).",
            markdown_table(same_house, {"estimate": money, "vs_median_area": lambda x: f"{x:+.1%}"}),
        ]
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")
    print(f"wrote {out}")


def run(sample=False, n_splits=N_SPLITS, drop_column=True):
    """Stage 6: run every analysis, save each table to reports/analysis*/ and write report.md."""
    cfg = load_config()
    training_cfg = cfg["training"]
    df, reports = load_data(sample)
    print(f"analysis on {len(df):,} houses, {n_splits} splits")

    metrics, test, train, models = repeated_splits(df, training_cfg, n_splits, drop_column)
    summary = summarize_splits(metrics)
    print(summary.to_string(index=False))
    save_dataframe(metrics, reports, "split_metrics.csv")
    save_dataframe(summary, reports, "summary.csv")

    estimate = f"estimate:{BEST_MODEL}"
    by_area = error_table(test, estimate, "area").sort_values("mape", ascending=False)
    by_price = error_table(test.assign(price_band=price_band(test[TARGET])), estimate, "price_band")
    by_use = error_table(test, estimate, "land_use_class")
    for name, table in [("errors_by_area", by_area), ("errors_by_price", by_price), ("errors_by_land_use", by_use)]:
        save_dataframe(table, reports, f"{name}.csv")

    decided = None
    if drop_column:
        importance = group_importance(train, test, training_cfg)
        decided = verdicts(drop_column_results(metrics), importance)
        print(decided.to_string(index=False))
        save_dataframe(importance, reports, "group_importance.csv")
        save_dataframe(decided, reports, "drop_column.csv")

    same_house = same_house_by_area(train, *models[f"{GROUP_FEATURE_SET}__{GROUP_MODEL}"])
    print(same_house.to_string(index=False))
    save_dataframe(same_house, reports, "same_house_by_area.csv")

    write_report(reports, len(df), n_splits, summary, by_area, by_price, by_use, decided, same_house)
    return summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Model reliability and feature-group analysis.")
    parser.add_argument("--sample", action="store_true", help="use the 10,000-house sample in data/sample/")
    parser.add_argument("--splits", type=int, default=N_SPLITS, help=f"number of random splits (default {N_SPLITS})")
    parser.add_argument("--skip-drop-column", action="store_true", help="skip the feature-group tests (much faster)")
    args = parser.parse_args()
    run(args.sample, args.splits, not args.skip_drop_column)


if __name__ == "__main__":
    main()
