"""
Retraining Pipeline
-------------------
1. Run feature drift detection.
2. If retraining is recommended (or forced), rebuild features and retrain models.

Run:
    python pipelines/retraining_pipeline.py
    # or
    python -m pipelines.retraining_pipeline
"""
import argparse

from loguru import logger

from src.ingestion.load_to_warehouse import main as ingest
from src.features.run_feature_pipeline import main as run_features
from src.features.change_point_detection import main as detect_change_points
from src.monitoring.drift_detector import run_drift_report
from src.models.churn_model import main as train_churn
from src.models.uplift_model import main as train_uplift


def main(force: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("RETRAINING PIPELINE")
    logger.info("=" * 60)

    should_retrain = force
    report = None

    if not force:
        try:
            logger.info("Running drift detection to decide on retraining...")
            report = run_drift_report()
            logger.info(
                "Drift report — drifted: {drifted}/{total} ({share:.1%}) | "
                "retraining_recommended: {retrain}".format(
                    drifted=report["drifted_features"],
                    total=report["total_features"],
                    share=report["drift_share"],
                    retrain=report["retraining_recommended"],
                )
            )
            should_retrain = report.get("retraining_recommended", False)
        except FileNotFoundError:
            logger.warning(
                "Feature export files not found for drift detection. "
                "Assuming initial training is required."
            )
            should_retrain = True

    if not should_retrain:
        logger.info("No retraining needed. Exiting.")
        return

    logger.info("Retraining triggered — rebuilding warehouse features and models...")

    logger.info("[1/4] Ingesting data into warehouse...")
    ingest()

    logger.info("[2/4] Running feature pipeline...")
    run_features()

    logger.info("[3/4] Detecting behavioral change points...")
    detect_change_points()

    logger.info("[4/4] Training models (churn + uplift)...")
    train_churn(do_train=True)
    train_uplift(do_train=True)

    logger.success("Retraining pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of drift status.",
    )
    args = parser.parse_args()
    main(force=args.force)

