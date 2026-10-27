"""Build the small demo dataset (data/sample/properties_sample.csv) and the shareable
copy of the full feature table from data/features/properties_features.csv.

Both copies drop the street address and deed book/page columns. The models never use
them, and leaving them out keeps the shared files from listing every home address in
the county next to its sale price. The sample also drops `parcel_id`.

The sample is what bin/quick_train.py trains on, so anyone can try the pipeline
without downloading the full data. It only needs to be rebuilt if the feature table
changes.

Usage (from the repository root):
    uv run python bin/make_sample.py                       # sample only
    uv run python bin/make_sample.py --shared OUTPUT_DIR   # sample + full shareable copy
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

SAMPLE_SIZE = 10_000
PRIVATE_COLUMNS = ["situs_street_num", "situs_street_name", "situs_zip", "deed_book", "deed_page"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shared", help="also write the full feature table, minus private columns, here")
    args = parser.parse_args()

    cfg = load_config()
    df = read_dataframe(resolve_path(cfg, "features_data"), "properties_features.csv")
    df = df.drop(columns=PRIVATE_COLUMNS)

    sample = df.sample(SAMPLE_SIZE, random_state=cfg["training"]["random_state"]).drop(columns=["parcel_id"])
    save_dataframe(sample, os.path.join(cfg["paths"]["root"], "data", "sample"), "properties_sample.csv")

    if args.shared:
        save_dataframe(df, os.path.join(args.shared, "data", "features"), "properties_features.csv")


if __name__ == "__main__":
    main()
