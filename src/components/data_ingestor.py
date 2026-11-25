"""Stage 1: download the raw data.

Downloads, each skipped if its file is already there:

- the Mecklenburg County CAMA parcel table, from the county's ArcGIS service in
  config/config.yaml (`mecklenburg`), to data/raw/merklenburg/Cama_Table_transformed.csv
- the Overture Maps places file (schools, clinics, parks, fire stations and so on) for
  the bounding box in config/config.yaml, to data/raw/NC parquet/nc_places.parquet
"""

import os
import json
import subprocess
import requests
import pandas as pd
from pathlib import Path
from src.utils.load_config import load_config

# The county's server refuses requests without a browser-like User-Agent (HTTP 403).
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; house-price-prediction pipeline)"}
MECKLENBURG_FILE = "merklenburg/Cama_Table_transformed.csv"


class data_ingestor:
    """Downloads the raw input files listed in config/config.yaml into data/raw/."""
    def __init__(self):
        """Load the config and remember the project root and raw-data folder."""
        self.cfg = load_config()
        self.ROOT_PATH = self.cfg["paths"]["root"]
        self.RAW_PATH = self.cfg['paths']['raw_data']

    def ingest(self):
        """Download the Mecklenburg CAMA table and the Overture places file (each
        skipped if already present), then page through each ArcGIS source in config
        `sources` (none are active while the project is Mecklenburg-only)."""
        raw = self.ROOT_PATH + self.RAW_PATH
        if self.cfg.get('mecklenburg'):
            self.fetch_mecklenburg(raw + MECKLENBURG_FILE, self.cfg['mecklenburg'])
        self.fetch_places(raw + 'NC parquet/nc_places.parquet', self.cfg['nc_places']['bbox'])
        for source in self.cfg.get('sources') or {}:
            pth = raw + source
            os.makedirs(pth, exist_ok=True)
            payload = self._fetch_all(source)
            out_file = os.path.join(pth, f"{source}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"{source}: {len(payload['features'])} features -> {out_file}")

    def fetch_mecklenburg(self, path, source_cfg):
        """Download every parcel's attributes from the county's ArcGIS `query` endpoint
        and save them as a CSV with the county's column names, the same layout as the
        county's CSV download. Dates come from the service as epoch milliseconds and
        are written as ISO dates (Eastern time), which data_preprocessing reads.
        Skipped if the file already exists. Returns the file path."""
        path = Path(path)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        rows, date_fields = [], []
        offset, page_size = 0, source_cfg['page_size']
        with requests.Session() as session:
            session.headers.update(HEADERS)
            while True:
                r = session.get(source_cfg['url'], timeout=120, params={
                    "where": source_cfg.get("where", "1=1"), "outFields": "*", "returnGeometry": "false", "f": "json",
                    "orderByFields": "objectid_1", "resultOffset": offset, "resultRecordCount": page_size,
                })
                r.raise_for_status()
                page = r.json()
                if "error" in page:
                    raise RuntimeError(f"ArcGIS error at offset {offset}: {page['error']}")
                if not date_fields:
                    date_fields = [f["name"] for f in page.get("fields", []) if f["type"] == "esriFieldTypeDate"]
                batch = [f["attributes"] for f in page.get("features", [])]
                rows.extend(batch)
                offset += len(batch)
                if offset % 50_000 < page_size:
                    print(f"mecklenburg: {offset:,} parcels downloaded")
                if not batch or not page.get("exceededTransferLimit", len(batch) == page_size):
                    break
        if not rows:
            raise RuntimeError(f"no parcels returned from {source_cfg['url']}; check the URL and `where` filter")
        df = pd.DataFrame(rows)
        for col in date_fields:
            df[col] = (pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
                       .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
        df.to_csv(path, index=False)
        print(f"mecklenburg: {len(df):,} parcels -> {path}")
        return path

    def fetch_places(self, path, bbox):
        """Download Overture Maps places inside `bbox` ("min_lon,min_lat,max_lon,max_lat")
        to the GeoParquet file at the given path, using the overturemaps command-line
        tool. Skipped if the file already exists. Returns the file path."""
        path = Path(path)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "overturemaps", "download",
            "--type=place",
            f"--bbox={bbox}",
            "-f", "geoparquet",
            "-o", str(path),
            # the STAC catalog can lag a new release and report no data for the bbox;
            # reading the S3 dataset directly doesn't depend on it
            "--no-stac",
        ], check=True)
        return path

    def _fetch_all(self, source):
        """Page through an ArcGIS FeatureServer `query` endpoint (resultOffset/resultRecordCount)
        until the full result set has been retrieved, and return a single combined payload."""
        source_cfg = self.cfg['sources'][source]
        page_size = source_cfg['page_size']
        params = {**self.cfg['params'], "resultRecordCount": page_size}
        features = []
        meta = None
        offset = 0

        while True:
            r = requests.get(source_cfg['url'], params={**params, "resultOffset": offset}, headers=HEADERS, timeout=60)
            r.raise_for_status()
            page = r.json()
            if meta is None:
                meta = {k: v for k, v in page.items() if k != "features"}
            batch = page.get("features", [])
            features.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)

        meta["features"] = features
        meta["exceededTransferLimit"] = False

        return meta


if __name__ == "__main__":
    data_ingestor().ingest()
