# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that estimates the 2023 county-assessed value (`total_value`) of residential parcels in Mecklenburg County, NC, from house details, location, neighborhood sales and accessibility features built from Overture Maps places. The research question: how much of the value comes from the house, the area and its access to education, health care, emergency services and activities, and how does the same house's value differ between areas? It is a student project by Tam Khong, Eric Toan and Ethan Wan. README.md documents every stage and function in detail.

## Course requirement: serverless

The project is graded on being fully serverless (S3, Athena, SageMaker, APIs). Work that relies on uploading data from a laptop, or on training or running models locally, loses points. The current code is local-only: the paths in `config.yaml` point at local folders, and the models were trained on a laptop. New work should move data to S3, querying to Athena, and training, analysis and prediction to SageMaker. Overture places is a public S3 dataset that Athena can query directly. The county CAMA data should be downloaded straight into S3 from SageMaker, not uploaded from a laptop. An API is not required; S3 + Athena + SageMaker covers the requirement.

## Commands

Run everything from this directory (the repo root, which holds `pyproject.toml`). Sessions sometimes start in the outer `house-price-prediction-main` wrapper folder, where commands fail. Modules must be run with `-m`, never as file paths, or `import src` fails.

This local copy is not a git repo. The GitHub repo is `tkhong-gif/house-price-prediction`; changes are copied into a clone to be pushed.

```bash
uv sync                                   # install (Python 3.14, uv.lock); add --extra xgboost for XGBoost
uv run pytest                             # all tests, ~15 s, sample data only
uv run pytest test/test_models.py::test_select_stages   # one test

uv run python bin/quick_train.py          # demo: train both model types on data/sample/ (~10 s)
uv run python bin/predict_example.py      # estimate predict.EXAMPLE_HOUSE; --demo, --parcel <id>
uv run python -m src.pipeline --sample    # no-download demo incl. 2-split analysis (~2 min)

uv run python -m src.pipeline                     # ingest -> preprocess -> features -> train -> evaluate -> analyze
uv run python -m src.pipeline --from train        # resume
uv run python -m src.pipeline --only evaluate,analyze   # finetune only runs via --only
uv run python -m src.components.<stage>           # one stage on its own
uv run python -m src.components.analysis --sample --splits 2 --skip-drop-column   # fast analysis options
```

No linter or formatter is configured. CI (`.github/workflows/tests.yml`) runs `uv sync`, `pytest`, `quick_train.py` and `predict_example.py --demo`.

Reference results (full data, Nov 2026), for judging changes: gradient boosting `with_neighborhood_price` has 5.1% MAPE, MAE $27,309, R² 0.967 (`reports/training_metrics.csv`). The analysis (5 splits, fixed settings) gives 5.2% against 16.7% for the baseline. The amenity groups (education, health care, emergency, activity, density) each add ≤1%.

Full-data runs are slow and print nothing until they finish. On the dev laptop (8 threads), `training` took about 2¼ hours and `analysis` with 5 splits several hours. Don't start them casually. Check for running `python` processes before assuming a run is finished, since models and reports are only written at the end.

## Architecture

**Stages communicate only through files.** Each module in `src/components/` reads the previous stage's CSV and writes its own. `src/pipeline/run_pipeline.py` defines `STAGES` (function, input files, hint if missing) and a `PRODUCER` map. It checks every input before the first stage starts, skipping inputs that an earlier stage in the same run will produce. All paths come from `config/config.yaml` through `load_config()`, which sets `paths.root` from the file location, plus `resolve_path(cfg, key)`.

Data flow: raw CAMA CSV + Overture parquet → `data_preprocessing` (residential filter, arms-length sale flag, `area` assignment: towns, Charlotte split into 12 parts by `median_split`, UNINC/STALLINGS merged into the nearest town) → `feature_engineering` → `data/features/properties_features.csv` → `training` → `output/models/*.joblib` + `reports/training_metrics.csv`.

**The leakage rule is central.** Sale-derived features (`SALE_DERIVED_FEATURES`: the neighborhood median $/sqft, sale count, `price_momentum`) are stored in the feature file computed on all rows. `load_training_data` drops them, and `training.split_with_aggregates` recomputes them from training rows only. Every consumer goes through it: training, evaluating, finetune and analysis. Any new feature derived from `sale_price` must be added to that list and recomputed the same way.

**Models are organized by (feature set × model type).**
- `training.FEATURE_SETS` has `with_neighborhood_price` (the one meant for estimates) and `without_neighborhood_price` (used to measure what location and amenities are worth).
- `training.MODEL_TYPES` is `["random_forest", "gradient_boosting"]`. Adding a type means updating `make_model`/`DEFAULT_PARAMS` (fixed settings used by `quick_train` and `analysis`) and `build_search_estimators` (the RandomizedSearchCV grid used by `training`). XGBoost is defined there but switched off.
- Models are trained on `log1p(total_value)`, and predictions are `expm1`'d back to dollars.
- One preprocessor is saved per feature set and shared by every model type in that set. So `finetune` writes to `output/models/finetuned/` rather than overwriting it.

**Which model gets used:** `predict.best_model_name` picks the trained model with the lowest RMSE in `reports/training_metrics.csv` (gradient boosting on the current full run). `evaluating` and `predict` skip models whose files don't exist.

**`analysis.py` trains its own models** with `DEFAULT_PARAMS` over repeated splits (split 0 = training's split). It doesn't load `output/models/`. Its feature-group tests require `FEATURE_GROUPS` to cover `FEATURE_SETS["without_neighborhood_price"]` exactly; a test enforces this. So a new feature must be added to a group. A group "matters" only if dropping it raises MAE ≥1% in most splits *and* shuffling it beats a random-noise column. Outputs go to `reports/analysis/`, or `reports/analysis_sample/` with `--sample`.

**`evaluating.py` rebuilds training's test split**, so it is only valid while `properties_features.csv` is unchanged since training. Retrain after rebuilding features.

## Data and privacy

- `data/raw/merklenburg/Cama_Table_transformed.csv` (the folder name is misspelled on purpose; keep it) contains owner names and mailing addresses. It is not shared: not on GitHub and not on the project's Google Drive. Never copy its personal columns into outputs, notebook outputs, the sample or commits. Never print owner columns; look up a single parcel's address only when asked.
- The raw file is Mecklenburg County's "Tax Parcels with CAMA Data". The `ingest` stage downloads it (`data_ingestor.fetch_mecklenburg`) from the county ArcGIS service in `config.yaml` (`mecklenburg`: `TaxParcel_camadata/FeatureServer/0`, ~450k rows, 2,000 per request; the server returns 403 without a browser-like User-Agent; object IDs start at ~20.6 million, not 1). Manual downloads are named `Cama_Table.csv` (98 comma-separated columns, the county's original names such as `parcelid`, `address`, `streetname`, `loccity`, `ownrlstnme`); rename to `Cama_Table_transformed.csv`. Sale dates come in three formats (Excel re-save `25/05/22 0:00`, county CSV `5/25/2022 0:00:00.000`, ISO from the service); `parse_sale_date` handles all three, and a wrong format silently empties every sale-derived feature. Rebuilding the features from a new download changes `properties_features.csv`, which invalidates the trained models and `evaluating`'s split, so retrain afterwards.
- `.gitignore` excludes `data/` (except `data/sample/`), `output/` and `reports/`, except the aggregate results (`training_metrics.csv`, `best_hyperparameters.json`, `reports/analysis*/`). `reports/validation_houses.csv` stays private because it has parcel IDs and sale prices.
- `data/sample/properties_sample.csv` (10k rows, no IDs or addresses) is the only data in the repo. It is rebuilt by `bin/make_sample.py`. Tests and CI depend on it.
- The full feature table and trained models (random forests ~1 GB each) are on Google Drive (link in the README) and too large for GitHub.
- `property_age` and the arms-length 10-year window use the current year, while the target is the 2023 assessment. Rebuilding features in a later year shifts them, so retrain afterwards.

## Project decisions

- No API or Dockerfile is planned; neither is required for the serverless grading. Don't add them unless asked.
- Places features use raw miles rather than decay scores (tree models split on thresholds). Health-care and activity are *shares* of `places_count` because raw counts were 0.91-correlated with density. Parks are inside `activity_share`, not a separate feature.
