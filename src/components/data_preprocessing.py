"""Stage 2: map the raw county records onto one schema, clean them and assign each house an area.
Only Mecklenburg is used; the loaders for other counties are commented out. Output:
data/processed/properties_clean.csv."""

import re
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.utils.common import resolve_path, save_dataframe
from src.utils.load_config import load_config

COMMON_COLUMNS = [
    "county", "parcel_id", "owner_name", "situs_street_num", "situs_street_name", "situs_city", "situs_zip",
    "sale_price", "sale_date", "deed_book", "deed_page", "land_value", "building_value", "total_value",
    "acreage", "land_use_class", "neighborhood_code", "year_built", "heated_area", "bedrooms", "bathrooms",
    "latitude", "longitude",
]

LAND_USE_MAP = {
    "R": "single_family",
    "100": "single_family",
    "SINGLE FAMILY RESIDENTIAL": "single_family",
    "SINGLE FAMILY RESIDENTIAL - ACREAGE": "single_family",
    "RESIDENTIAL": "single_family",
    "TOWN HOUSE  SFR": "townhouse",
    "TOWNHOUSE": "townhouse",
    "CONDOMINIUM": "condo",
    "311": "condo",
    "V": "vacant_land",
    "N": "vacant_land",
    "VACANT": "vacant_land",
    "VACANT LAND": "vacant_land",
}

MIN_SALE_YEAR = 1900
ARMS_LENGTH_MIN_PRICE = 10_000
ARMS_LENGTH_MAX_PRICE = 20_000_000
ARMS_LENGTH_LOOKBACK_YEARS = 10

NON_INDIVIDUAL_OWNER_KEYWORDS = [
    "LLC", "L L C", "INC", "INCORPORATED", "CORP", " CO ", " CO,", " CO.", "COMPANY",
    "LP", "LLP", "LTD", "TRUST", "ESTATE OF", "BANK", "CHURCH", "ASSOC", "ASSN",
    "PARTNERSHIP", "PROPERTIES", "INVESTMENTS", "HOLDINGS", "MGMT", "MANAGEMENT",
    "FOUNDATION", "MINISTR", "HOMEOWNERS", " HOA", "AUTHORITY", "DEVELOPMENT",
    "BUILDERS", "CONSTRUCTION", "REALTY", "ENTERPRISES", "VENTURES", "CAPITAL",
    "UNIVERSITY", "HOSPITAL", "SCHOOL", "GOVERNMENT", "CITY OF", "COUNTY OF", "STATE OF",
    "UNITED STATES", "HABITAT FOR HUMANITY",
]
NON_INDIVIDUAL_OWNER_PATTERN = "|".join(re.escape(k) for k in NON_INDIVIDUAL_OWNER_KEYWORDS)

RESIDENTIAL_LAND_USE_CLASSES = ["single_family", "condo", "townhouse"]

MIN_HEATED_AREA = 300
MAX_HEATED_AREA = 10_000

MERGE_INTO_NEAREST_CITY = ["UNINC", "STALLINGS"]
NEAREST_CITY_NEIGHBORS = 15

SPLIT_CITY = "CHARLOTTE"
MAX_AREA_SIZE = 30_000


NUMERIC_COLUMNS = [
    "situs_street_num", "situs_zip", "sale_price", "land_value", "building_value",
    "total_value", "acreage", "year_built", "heated_area", "bedrooms", "bathrooms",
    "latitude", "longitude",
]





def load_mecklenburg(raw_path):
    """Read the Mecklenburg County CAMA table from data/raw/merklenburg/ as a
    DataFrame, with the county's original column names."""
    path = raw_path + "merklenburg/Cama_Table_transformed.csv"
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    european = header.count(";") > header.count(",")
    return pd.read_csv(
        path, encoding="utf-8-sig", low_memory=False,
        sep=";" if european else ",", decimal="," if european else ".",
    )


SALE_DATE_FORMATS = ["%d/%m/%y %H:%M", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%d"]


def parse_sale_date(values):
    """Parse sale dates written in any of SALE_DATE_FORMATS; anything else becomes NaT.
    Each value takes the first format that fits, so the original European format
    gives exactly the same dates as before."""
    values = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[us]")
    for fmt in SALE_DATE_FORMATS:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed[missing] = pd.to_datetime(values[missing], format=fmt, errors="coerce")
    return parsed


def meck_to_common(df):
    """Map the Mecklenburg columns onto the shared schema (COMMON_COLUMNS): owner
    name from last + first name, bathrooms as full baths + 0.5 per half bath, and
    latitude/longitude from the source's swapped x/y columns."""
    out = pd.DataFrame({
        "county": "Mecklenburg",
        "parcel_id": df["parcelid"],
        "owner_name": (df["ownrlstnme"].fillna("") + " " + df["ownrfrstnme"].fillna("")).str.strip(),
        "situs_street_num": df["streetnumber"],
        "situs_street_name": df["streetname"],
        "situs_city": df["loccity"],
        "situs_zip": pd.NA,
        "sale_price": df["saleprice"],
        "sale_date": parse_sale_date(df["saledate"]),
        "deed_book": df["deed_book"],
        "deed_page": df["deed_page"],
        "land_value": df["totlandval"],
        "building_value": df["totalbldgval"],
        "total_value": df["totalvalue"],
        "acreage": df["gisacres"],
        "land_use_class": df["landuse_description"],
        "neighborhood_code": df["neighborhood"],
        "year_built": df["yearbuilt"],
        "heated_area": df["heatedarea"],
        "bedrooms": df["bedrooms"],
        "bathrooms": df["fullbath"] + 0.5 * df["halfbath"],
        "latitude": df["xcoord"],
        "longitude": df["ycoord"],
    })
    return out[COMMON_COLUMNS]




def nc_places_map(path,cfg):
    """Reduce the Overture places GeoParquet at `path` to four columns (lon, lat,
    name, category), save it as data/processed/nc_places_map.csv and return its path."""
    poi = gpd.read_parquet(path)
    df = pd.DataFrame({
        "lon": poi.geometry.x,
        "lat": poi.geometry.y,
        "name": poi["names"].str["primary"],
        "cat": poi["taxonomy"].str["primary"],
    })
    return save_dataframe(df, resolve_path(cfg, "processed_data"), "nc_places_map.csv")




def merge_all_counties(raw_path):
    """Load every county and stack them into one table on the shared schema. Only
    Mecklenburg is currently included."""
    return pd.concat(
        [
            meck_to_common(load_mecklenburg(raw_path)),
            
        ],
        ignore_index=True,
    )


def normalize_land_use(series):
    """Map raw land_use_class values onto LAND_USE_MAP; unrecognized non-null values
    become "other", missing values stay missing."""
    mapped = series.map(LAND_USE_MAP)
    fallback = series.where(series.isna(), "other")
    return mapped.fillna(fallback)


def sale_price_outlier_mask(sale_price):
    """Flags sale_price values outside 1.5*IQR on a log1p scale (real-estate prices are
    right-skewed, so IQR on the raw scale would flag almost every high-end sale)."""
    valid = sale_price.notna() & (sale_price > 0)
    log_price = np.log1p(sale_price[valid])
    q1, q3 = log_price.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = pd.Series(False, index=sale_price.index)
    is_outlier.loc[valid] = (log_price < lower) | (log_price > upper)
    return is_outlier


def is_individual_owner(owner_name):
    """True for owners that look like a person: no business or institution keyword
    (NON_INDIVIDUAL_OWNER_KEYWORDS, e.g. LLC, TRUST, CHURCH) in the name. Blank names
    count as individuals, so those rows are kept."""
    upper = owner_name.fillna("").str.upper()
    is_entity = upper.str.contains(NON_INDIVIDUAL_OWNER_PATTERN, regex=True)
    is_blank = upper.str.strip() == ""
    return is_blank | ~is_entity


def clean_properties(df):
    """Clean the combined property table: drop duplicates, blank impossible values and flag normal market sales (`arms_length_sale`).
    Keep only single-family homes, condos and townhouses of a plausible size, and remove
    business and institution owners."""
    df = df.drop_duplicates().copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    current_year = datetime.now().year
    bad_date = (df["sale_date"].dt.year < MIN_SALE_YEAR) | (df["sale_date"].dt.year > current_year)
    df.loc[bad_date, "sale_date"] = pd.NaT

    df.loc[df["sale_price"] == 0, "sale_price"] = pd.NA
    df.loc[sale_price_outlier_mask(df["sale_price"]), "sale_price"] = pd.NA

    df.loc[(df["year_built"] == 0) | (df["year_built"] > current_year), "year_built"] = pd.NA

    recent_cutoff = pd.Timestamp(year=current_year - ARMS_LENGTH_LOOKBACK_YEARS, month=1, day=1)
    df["arms_length_sale"] = (
        df["sale_price"].between(ARMS_LENGTH_MIN_PRICE, ARMS_LENGTH_MAX_PRICE)
        & (df["sale_date"] >= recent_cutoff)
    )

    df["land_use_class"] = normalize_land_use(df["land_use_class"])
    df = df[df["land_use_class"].isin(RESIDENTIAL_LAND_USE_CLASSES)]

    bad_heated_area = df["heated_area"].notna() & ~df["heated_area"].between(MIN_HEATED_AREA, MAX_HEATED_AREA)
    df = df[~bad_heated_area]

    df = df[is_individual_owner(df["owner_name"])].copy()
    df = df.drop(columns=["owner_name", "situs_zip"])

    return df


def nearest_label(train_df, label_col, query_df, k=NEAREST_CITY_NEIGHBORS):
    """Majority `label_col` among each query row's k nearest train rows (haversine on lat/lon)."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="haversine", algorithm="ball_tree")
    knn.fit(np.radians(train_df[["latitude", "longitude"]]), train_df[label_col])
    return knn.predict(np.radians(query_df[["latitude", "longitude"]]))


def median_split(nbhds, max_size):
    """Recursively halve neighborhoods (index: neighborhood_code; columns: latitude,
    longitude, n) by house count along their wider axis, until each part has at most
    max_size houses. Returns a list of parts."""
    if nbhds["n"].sum() <= max_size or len(nbhds) == 1:
        return [nbhds]
    lat_span = nbhds["latitude"].max() - nbhds["latitude"].min()
    lon_span = (nbhds["longitude"].max() - nbhds["longitude"].min()) * np.cos(np.radians(nbhds["latitude"].mean()))
    nbhds = nbhds.sort_values("latitude" if lat_span >= lon_span else "longitude")
    first_half = nbhds["n"].cumsum().shift(fill_value=0) < nbhds["n"].sum() / 2
    return median_split(nbhds[first_half], max_size) + median_split(nbhds[~first_half], max_size)


def assign_area(df):
    """Add `area`: the parcel's town, with MERGE_INTO_NEAREST_CITY parcels moved to the
    nearest town and Charlotte split into CHARLOTTE_1..N (numbered north to south)."""
    df = df.copy()
    df["area"] = df["situs_city"]
    merge = df["area"].isin(MERGE_INTO_NEAREST_CITY)
    df.loc[merge, "area"] = nearest_label(df[~merge], "area", df[merge])

    in_city = df["area"] == SPLIT_CITY
    has_nbhd = in_city & df["neighborhood_code"].notna()
    nbhds = df[has_nbhd].groupby("neighborhood_code").agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean"), n=("parcel_id", "size"),
    )
    parts = sorted(median_split(nbhds, MAX_AREA_SIZE), key=lambda part: -part["latitude"].mean())
    nbhd_area = {code: f"{SPLIT_CITY}_{i}" for i, part in enumerate(parts, 1) for code in part.index}
    df.loc[has_nbhd, "area"] = df.loc[has_nbhd, "neighborhood_code"].map(nbhd_area)

    no_nbhd = in_city & ~has_nbhd
    if no_nbhd.any():
        df.loc[no_nbhd, "area"] = nearest_label(df[has_nbhd], "area", df[no_nbhd])
    return df


def run():
    """Stage 2: write the places map, then load, clean and assign areas to the county
    records and save data/processed/properties_clean.csv."""
    cfg = load_config()
    raw_path = resolve_path(cfg, "raw_data")
    processed_path = resolve_path(cfg, "processed_data")
    nc_places_map(raw_path+ "NC parquet/nc_places.parquet",cfg)
    combined = merge_all_counties(raw_path)
    clean = assign_area(clean_properties(combined))
    save_dataframe(clean, processed_path, "properties_clean.csv")
    return clean


if __name__ == "__main__":
    run()
