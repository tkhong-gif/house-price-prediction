"""Tests for src/components/data_preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.components.data_preprocessing import (
    COMMON_COLUMNS, assign_area, clean_properties, is_individual_owner, median_split,
    normalize_land_use, parse_sale_date, sale_price_outlier_mask,
)


def test_parse_sale_date_reads_every_known_format():
    raw = pd.Series(["25/05/22 0:00", "5/25/2022 0:00:00.000", "2022-05-25", None, "not a date"])
    parsed = parse_sale_date(raw)
    assert (parsed[:3] == pd.Timestamp("2022-05-25")).all()
    assert parsed[3:].isna().all()


def make_properties(n=6, **overrides):
    """A small table on the shared schema, with ordinary values that pass every filter."""
    df = pd.DataFrame({
        "county": "Mecklenburg",
        "parcel_id": range(1, n + 1),
        "owner_name": "SMITH JOHN",
        "situs_street_num": 100,
        "situs_street_name": "MAIN ST",
        "situs_city": "CHARLOTTE",
        "situs_zip": pd.NA,
        "sale_price": 300_000.0,
        "sale_date": pd.Timestamp("2022-06-01"),
        "deed_book": 1,
        "deed_page": 1,
        "land_value": 100_000,
        "building_value": 200_000,
        "total_value": 300_000,
        "acreage": 0.25,
        "land_use_class": "SINGLE FAMILY RESIDENTIAL",
        "neighborhood_code": "A1",
        "year_built": 2000,
        "heated_area": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "latitude": np.linspace(35.1, 35.3, n),
        "longitude": -80.8,
    })
    for col, values in overrides.items():
        df[col] = values
    return df[COMMON_COLUMNS]


def test_normalize_land_use_maps_known_values_and_keeps_missing():
    out = normalize_land_use(pd.Series(["CONDOMINIUM", "WAREHOUSE", None]))
    assert out.tolist()[:2] == ["condo", "other"]
    assert pd.isna(out.iloc[2])


def test_sale_price_outlier_mask_flags_extremes_only():
    prices = pd.Series([250_000.0] * 20 + [300_000.0] * 20 + [1.0, 1e10, np.nan, 0.0])
    mask = sale_price_outlier_mask(prices)
    assert mask.iloc[-4] and mask.iloc[-3]
    assert not mask.iloc[:40].any()
    assert not mask.iloc[-2] and not mask.iloc[-1]  # missing and zero are not "outliers"


@pytest.mark.parametrize("name, individual", [
    ("SMITH JOHN", True), ("", True), ("ACME HOLDINGS LLC", False), ("FIRST BAPTIST CHURCH", False),
    ("CITY OF CHARLOTTE", False),
])
def test_is_individual_owner(name, individual):
    assert is_individual_owner(pd.Series([name])).iloc[0] == individual


def test_clean_properties_filters_rows_and_drops_private_columns():
    df = make_properties(
        land_use_class=["SINGLE FAMILY RESIDENTIAL", "CONDOMINIUM", "VACANT", "TOWNHOUSE",
                        "SINGLE FAMILY RESIDENTIAL", "SINGLE FAMILY RESIDENTIAL"],
        heated_area=[2000, 1200, 2000, 1500, 50_000, np.nan],
        owner_name=["SMITH JOHN", "DOE JANE", "LEE AMY", "ABC PROPERTIES LLC", "KIM BO", "PARK JI"],
    )
    clean = clean_properties(df)
    # vacant land, an LLC owner and a 50,000 sq ft "house" are removed; missing heated area is kept
    assert clean["parcel_id"].tolist() == [1, 2, 6]
    assert set(clean["land_use_class"]) <= {"single_family", "condo", "townhouse"}
    assert "owner_name" not in clean and "situs_zip" not in clean


def test_clean_properties_blanks_impossible_values():
    df = make_properties(
        n=3,
        year_built=[0, 3000, 1990],
        sale_price=[0.0, 300_000.0, 300_000.0],
        sale_date=pd.to_datetime(["2022-01-01", "1850-01-01", "2022-01-01"]),
    )
    clean = clean_properties(df).set_index("parcel_id")
    assert clean["year_built"].isna().tolist() == [True, True, False]
    assert pd.isna(clean.loc[1, "sale_price"])
    assert pd.isna(clean.loc[2, "sale_date"])
    assert clean["arms_length_sale"].tolist() == [False, False, True]


def test_median_split_respects_max_size_and_keeps_every_neighborhood():
    rng = np.random.default_rng(0)
    nbhds = pd.DataFrame({
        "latitude": rng.uniform(35, 35.5, 200),
        "longitude": rng.uniform(-81, -80.5, 200),
        "n": rng.integers(50, 500, 200),
    }, index=[f"N{i}" for i in range(200)])
    parts = median_split(nbhds, 5_000)
    assert all(p["n"].sum() <= 5_000 or len(p) == 1 for p in parts)
    assert sorted(code for p in parts for code in p.index) == sorted(nbhds.index)


def test_assign_area_merges_uninc_and_splits_charlotte():
    n = 40
    df = make_properties(n=n)
    df["situs_city"] = ["CHARLOTTE"] * 30 + ["MATTHEWS"] * 9 + ["UNINC"]
    df["neighborhood_code"] = [f"N{i % 6}" for i in range(n)]
    df["latitude"] = np.r_[np.linspace(35.0, 35.4, 30), np.full(9, 35.6), 35.6]
    df["longitude"] = np.r_[np.full(30, -80.8), np.full(9, -80.6), -80.6]
    out = assign_area(df)
    assert out.loc[39, "area"] == "MATTHEWS"  # the UNINC parcel sits among Matthews parcels
    assert out.loc[:29, "area"].str.startswith("CHARLOTTE").all()
