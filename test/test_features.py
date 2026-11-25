"""Tests for src/components/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from src.components.feature_engineering import (
    MAX_DISTANCE_MILES, MOMENTUM_START, _growth_by, add_neighborhood_aggregates, add_price_momentum,
    add_price_per_sqft, label_emergency_departments, nearest_distance, nearest_place_miles, places_within_miles,
)


def test_nearest_distance_matches_known_distance():
    # one degree of latitude is about 69.1 miles
    dist, idx = nearest_distance([35.0], [-80.0], [36.0, 40.0], [-80.0, -80.0])
    assert dist[0] == pytest.approx(69.1, abs=0.2)
    assert idx[0] == 0


def test_nearest_place_miles_caps_distance_and_keeps_missing_coordinates():
    houses = pd.DataFrame({"latitude": [35.0, np.nan], "longitude": [-80.0, -80.0]})
    places = pd.DataFrame({"lat": [45.0], "lon": [-80.0], "cat": ["school"]})
    out = nearest_place_miles(houses, places, ["school"])
    assert out.iloc[0] == MAX_DISTANCE_MILES
    assert pd.isna(out.iloc[1])


def test_places_within_miles_counts_by_category():
    houses = pd.DataFrame({"latitude": [35.0], "longitude": [-80.0]})
    places = pd.DataFrame({
        "lat": [35.0, 35.01, 35.02, 36.0], "lon": [-80.0] * 4, "cat": ["park", "park", "gym", "park"],
    })
    assert places_within_miles(houses, places, 5).iloc[0] == 3
    assert places_within_miles(houses, places, 5, ["park"]).iloc[0] == 2


def test_label_emergency_departments_by_name_not_category():
    places = pd.DataFrame({
        "lat": [35.0] * 4, "lon": [-80.0] * 4,
        "name": ["Atrium Health Pineville", "Uptown Emergency Department", "Dr. Smith", "Paws Emergency Room"],
        "cat": ["hospital", "travel_service", "emergency_department", "veterinarian"],
    })
    out = label_emergency_departments(places)["cat"].tolist()
    assert out == ["emergency_department_verified", "emergency_department_verified",
                   "emergency_department", "veterinarian"]


def test_add_price_per_sqft_only_for_arms_length_sales():
    df = pd.DataFrame({
        "sale_price": [300_000, 300_000, 300_000], "heated_area": [1500, 1500, 0],
        "arms_length_sale": [True, False, True],
    })
    out = add_price_per_sqft(df)
    assert out["price_per_sqft"].iloc[0] == 200
    assert out["price_per_sqft"].iloc[1:].isna().all()


def test_neighborhood_aggregates_use_only_source_rows():
    df = pd.DataFrame({"county": "M", "neighborhood_code": ["A", "A", "A"], "price_per_sqft": [100.0, 200.0, 900.0]})
    out = add_neighborhood_aggregates(df, source=df.iloc[:2])
    assert out["neighborhood_median_price_per_sqft"].tolist() == [150.0] * 3
    assert out["neighborhood_sale_count"].tolist() == [2] * 3


def test_growth_by_recovers_known_rate():
    t = np.linspace(0, 3, 40)
    sales = pd.DataFrame({"g": "x", "t": t, "log_ppsf": np.log(100) + np.log(1.2) * t})
    out = _growth_by(sales, ["g"])
    assert out["growth"].iloc[0] == pytest.approx(0.2)
    assert out["sales"].iloc[0] == 40


def test_price_momentum_falls_back_to_area_for_small_neighborhoods():
    rng = np.random.default_rng(0)
    dates = MOMENTUM_START + pd.to_timedelta(rng.uniform(0, 3 * 365, 60), unit="D")
    years = (dates - MOMENTUM_START).days / 365.25
    df = pd.DataFrame({
        "county": "M",
        "neighborhood_code": ["BIG"] * 50 + ["SMALL"] * 10,
        "area": "AREA_1",
        "sale_date": dates,
        "price_per_sqft": 100 * 1.1 ** years,
    })
    out = add_price_momentum(df)
    assert out["price_momentum"].round(6).nunique() == 1  # SMALL uses the area trend, same 10%
    assert out["price_momentum"].iloc[0] == pytest.approx(0.1)
