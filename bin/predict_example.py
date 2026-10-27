"""Estimate the assessed value of one example house.

By default it uses the full trained models in output/models/ (from training.py, or
downloaded from the project's Google Drive). With --demo it uses the small model
bin/quick_train.py saves to output/models/demo/, so it works right after the
get-started demo without downloading anything.

Usage (from the repository root):
    uv run python bin/predict_example.py           # full models
    uv run python bin/predict_example.py --demo    # demo model from quick_train.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components import predict

# Feature names and meanings are listed in the README (Feature engineering section).
# Any feature left out is filled with a typical value, as in training.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="use the demo model from bin/quick_train.py")
    args = parser.parse_args()

    if args.demo:
        # predict.load_model reads from the configured models folder, so point it at demo/.
        original = predict.load_artifact
        predict.load_artifact = lambda path, filename: original(path + "demo/", filename)
        feature_sets = ["with_neighborhood_price"]
    else:
        feature_sets = list(predict.FEATURE_SETS)

    print("Example house: 2 bed, 1 bath, 1,017 sq ft, built 1950, CHARLOTTE_1")
    for feature_set in feature_sets:
        try:
            value = predict.predict_total_value(EXAMPLE_HOUSE, feature_set=feature_set)
        except FileNotFoundError:
            hint = "run bin/quick_train.py first" if args.demo else "download the models (see README) or use --demo"
            sys.exit(f"model files not found: {hint}")
        print(f"  {feature_set}: ${value:,.0f}")


if __name__ == "__main__":
    main()
