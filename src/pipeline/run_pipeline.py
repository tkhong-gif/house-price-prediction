"""Run the pipeline stages in order: ingest, preprocess, features, train, evaluate and analyze (finetune only when asked).
Each stage's input files are checked before the run starts. Usage: uv run python -m
src.pipeline [--from STAGE | --only A,B | --sample]."""

import argparse
import os
import sys
import time

from src.utils.common import resolve_path
from src.utils.load_config import load_config


def _ingest():
    """Download the Overture places file (skipped if present)."""
    from src.components.data_ingestor import data_ingestor
    data_ingestor().ingest()


def _preprocess():
    """Clean the county records and assign areas."""
    from src.components import data_preprocessing
    data_preprocessing.run()


def _features():
    """Build the feature table."""
    from src.components import feature_engineering
    feature_engineering.run()


def _train():
    """Tune and train every model (the slow stage)."""
    from src.components import training
    training.run()


def _evaluate():
    """10-house validation of the trained models."""
    from src.components import evaluating
    evaluating.run()


def _analyze():
    """Baseline, repeated splits, error breakdowns and feature-group tests."""
    from src.components import analysis
    analysis.run()


def _finetune():
    """Retrain the best model on the current features."""
    from src.components import finetune
    finetune.run()


STAGES = {
    "ingest": (_ingest, [], ""),
    "preprocess": (_preprocess, [
        ("raw_data", "merklenburg/Cama_Table_transformed.csv"),
        ("raw_data", "NC parquet/nc_places.parquet"),
    ], "run the ingest stage, which downloads both files (see README, data_ingestor.py)"),
    "features": (_features, [
        ("processed_data", "properties_clean.csv"), ("processed_data", "nc_places_map.csv"),
    ], "run the preprocess stage first"),
    "train": (_train, [("features_data", "properties_features.csv")],
              "run the features stage, or download the feature table (see README, Full data)"),
    "evaluate": (_evaluate, [
        ("features_data", "properties_features.csv"),
        ("models", "with_neighborhood_price_preprocessor.joblib"),
        ("models", "with_neighborhood_price_random_forest.joblib"),
    ], "run the train stage, or download the models (see README, Full data)"),
    "analyze": (_analyze, [("features_data", "properties_features.csv")],
                "run the features stage, or download the feature table (see README, Full data)"),
    "finetune": (_finetune, [
        ("features_data", "properties_features.csv"),
        ("reports", "training_metrics.csv"), ("reports", "best_hyperparameters.json"),
    ], "run the train stage first"),
}
SAMPLE_SPLITS = 2
DEFAULT_STAGES = ["ingest", "preprocess", "features", "train", "evaluate", "analyze"]
PRODUCER = {"preprocess": "ingest", "features": "preprocess", "train": "features",
            "evaluate": "train", "analyze": "features", "finetune": "train"}


def missing_inputs(cfg, stage):
    """Input files of `stage` that don't exist yet."""
    return [os.path.join(resolve_path(cfg, key), name) for key, name in STAGES[stage][1]
            if not os.path.exists(os.path.join(resolve_path(cfg, key), name))]


def run_sample():
    """The no-download demo: train on the sample, estimate the example house, and run
    the analysis on the sample."""
    from src.components import analysis, predict
    sys.path.insert(0, os.path.join(load_config()["paths"]["root"], "bin"))
    import quick_train

    steps = [
        ("quick_train", quick_train.main),
        ("predict", lambda: print(f"example house: ${predict.predict_total_value(predict.EXAMPLE_HOUSE, models_dir='demo/'):,.0f}")),
        ("analyze", lambda: analysis.run(sample=True, n_splits=SAMPLE_SPLITS)),
    ]
    for name, step in steps:
        _timed(name, step)


def _timed(name, step):
    """Run one step with a header and its duration."""
    print(f"\n=== {name} ===")
    start = time.perf_counter()
    step()
    print(f"=== {name} finished in {time.perf_counter() - start:,.0f}s ===")


def select_stages(start=None, only=None):
    """The stages to run: `only` (comma-separated), or the default stages from `start` on."""
    if only:
        stages = [s.strip() for s in only.split(",")]
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}, expected some of {list(STAGES)}")
        return stages
    if start is None:
        return DEFAULT_STAGES
    if start not in DEFAULT_STAGES:
        raise SystemExit(f"--from must be one of {DEFAULT_STAGES}")
    return DEFAULT_STAGES[DEFAULT_STAGES.index(start):]


def run(stages):
    """Run `stages` in order. Every stage's inputs are checked before the first one
    starts, unless an earlier stage in the run writes them."""
    cfg = load_config()
    for i, stage in enumerate(stages):
        if PRODUCER.get(stage) in stages[:i]:
            continue
        missing = missing_inputs(cfg, stage)
        if missing:
            raise SystemExit(f"cannot run {stage!r}, missing:\n  " + "\n  ".join(missing) + f"\nhint: {STAGES[stage][2]}")
    for stage in stages:
        _timed(stage, STAGES[stage][0])


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the pipeline stages in order.")
    parser.add_argument("--from", dest="start", help=f"start at this stage, one of {DEFAULT_STAGES}")
    parser.add_argument("--only", help=f"comma-separated stages to run, from {list(STAGES)}")
    parser.add_argument("--sample", action="store_true", help="no-download demo on data/sample/")
    args = parser.parse_args()
    if args.sample:
        run_sample()
    else:
        run(select_stages(args.start, args.only))


if __name__ == "__main__":
    main()
