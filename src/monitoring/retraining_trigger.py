"""
Retraining Trigger
------------------

Small helper module that:
- Runs the drift detector
- Decides whether to trigger the retraining_pipeline

Intended usage:
    python -m src.monitoring.retraining_trigger
or to force retraining regardless of drift:
    python -m src.monitoring.retraining_trigger --force
"""
import argparse

from loguru import logger

from src.monitoring.drift_detector import run_drift_report
from pipelines.retraining_pipeline import main as run_retraining_pipeline


def should_retrain(force: bool = False) -> bool:
    """
    Return True if retraining should be triggered.

    - If force=True, always returns True.
    - Otherwise, uses run_drift_report() and its 'retraining_recommended' flag.
    """
    if force:
        logger.info("Force flag set — retraining will be triggered.")
        return True

    try:
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
        return bool(report.get("retraining_recommended", False))
    except FileNotFoundError as e:
        logger.warning(
            "Feature export files not found for drift detection (%s). "
            "Assuming initial training is required.", e
        )
        return True
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return False


def main(force: bool = False) -> None:
    """
    Entry point for CLI / schedulers.

    - Checks drift (or respects --force)
    - If retraining is needed, calls the retraining pipeline.
    """
    logger.info("=" * 60)
    logger.info("RETRAINING TRIGGER")
    logger.info("=" * 60)

    if not should_retrain(force=force):
        logger.info("No retraining needed. Exiting.")
        return

    logger.info("Retraining required — invoking retraining pipeline...")
    run_retraining_pipeline(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of drift status.",
    )
    args = parser.parse_args()
    main(force=args.force)

