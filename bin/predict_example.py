"""Estimate the assessed value of one house.

By default it estimates predict.EXAMPLE_HOUSE (edit that dict in src/components/predict.py
to try a different house) with the full trained models in output/models/ (from
training.py, or downloaded from the project's Google Drive). With --parcel it uses a
real house from the feature table instead, with all 20 features filled in and its actual
assessed value shown for comparison. With --demo it uses the small model
bin/quick_train.py saves to output/models/demo/, so it works right after the get-started
demo without downloading anything.

Usage (from the repository root):
    uv run python bin/predict_example.py                     # the example house, full models
    uv run python bin/predict_example.py --parcel 22339119   # a real house by parcel id
    uv run python bin/predict_example.py --demo              # demo model from quick_train.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.components import predict
from src.components.training import FEATURE_SETS, TARGET
from src.utils.common import resolve_path
from src.utils.load_config import load_config


def describe(house):
    """One line summing up a house, built from its own values."""
    def fmt(key, text):
        value = house.get(key)
        return text.format(value) if value is not None and not pd.isna(value) else None

    parts = [
        fmt("land_use_class", "{}"),
        fmt("bedrooms", "{:g} bed"),
        fmt("bathrooms", "{:g} bath"),
        fmt("heated_area", "{:,.0f} sq ft"),
        fmt("year_built", "built {:.0f}"),
        fmt("area", "{}"),
    ]
    return ", ".join(p for p in parts if p)


def load_parcel(parcel_id):
    """(features dict, actual assessed value) of one house in the feature table."""
    path = os.path.join(resolve_path(load_config(), "features_data"), "properties_features.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: download the feature table (see README, Full data)")
    columns = sorted(set(FEATURE_SETS["with_neighborhood_price"]) | {"parcel_id", TARGET})
    table = pd.read_csv(path, usecols=columns)
    rows = table[table["parcel_id"] == parcel_id]
    if rows.empty:
        sys.exit(f"parcel {parcel_id} is not in {path}")
    row = rows.iloc[0]
    return row.drop(["parcel_id", TARGET]).to_dict(), float(row[TARGET])


def main():
    """Print one house's estimate from every trained model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="use the demo model from bin/quick_train.py")
    parser.add_argument("--parcel", type=float, help="estimate this real house from the feature table")
    args = parser.parse_args()

    models_dir = "demo/" if args.demo else ""
    feature_sets = ["with_neighborhood_price"] if args.demo else list(predict.FEATURE_SETS)
    house, actual = (load_parcel(args.parcel) if args.parcel else (predict.EXAMPLE_HOUSE, None))

    print(f"House: {describe(house)}")
    if actual is not None:
        print(f"  actual assessed value: ${actual:,.0f}")
    found = 0
    for feature_set in feature_sets:
        for model_name in predict.MODEL_TYPES:
            try:
                value = predict.predict_total_value(
                    house, feature_set=feature_set, model_name=model_name, models_dir=models_dir,
                )
            except FileNotFoundError:
                print(f"  {feature_set} {model_name}: not trained yet")
                continue
            found += 1
            error = f"  ({value / actual - 1:+.1%})" if actual else ""
            print(f"  {feature_set} {model_name}: ${value:,.0f}{error}")
    if not found:
        hint = "run bin/quick_train.py first" if args.demo else "download the models (see README) or use --demo"
        sys.exit(f"model files not found: {hint}")


if __name__ == "__main__":
    main()
