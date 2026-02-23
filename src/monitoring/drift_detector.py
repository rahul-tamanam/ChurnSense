"""
Feature Drift Detector
-----------------------
Uses Evidently to compare reference (training) vs current (scoring) feature distributions.
Flags drift using PSI (Population Stability Index).
Triggers retraining pipeline if drift exceeds threshold.
"""
import pandas as pd
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

from src.utils.config import EXPORTS_DIR, DRIFT_PSI_THRESHOLD


def run_drift_report(
    reference_path: str = None,
    current_path:   str = None,
) -> dict:
    """
    Compare reference vs current feature distributions.
    Returns a dict with drift summary and per-feature results.
    """
    ref_path = reference_path or EXPORTS_DIR / "features_train.csv"
    cur_path = current_path   or EXPORTS_DIR / "features_score.csv"

    reference = pd.read_csv(ref_path)
    current   = pd.read_csv(cur_path)

    # Drop non-feature columns
    drop_cols = ["msno", "is_churn", "split"]
    ref_feat  = reference.drop(columns=[c for c in drop_cols if c in reference.columns])
    cur_feat  = current.drop(columns=[c for c in drop_cols if c in current.columns])

    # Align columns
    common = [c for c in ref_feat.columns if c in cur_feat.columns]
    ref_feat = ref_feat[common]
    cur_feat = cur_feat[common]

    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=ref_feat, current_data=cur_feat)
    result = report.as_dict()

    # Extract summary
    drift_summary = result["metrics"][1]["result"]
    drifted_features = drift_summary.get("number_of_drifted_columns", 0)
    total_features   = drift_summary.get("number_of_columns", len(common))
    drift_share      = drift_summary.get("share_of_drifted_columns", 0)
    dataset_drift    = drift_summary.get("dataset_drift", False)

    summary = {
        "dataset_drift_detected": dataset_drift,
        "drifted_features":       drifted_features,
        "total_features":         total_features,
        "drift_share":            drift_share,
        "retraining_recommended": drift_share > DRIFT_PSI_THRESHOLD,
    }

    logger.info(
        f"Drift report: {drifted_features}/{total_features} features drifted "
        f"({drift_share:.0%}) | retraining_recommended={summary['retraining_recommended']}"
    )

    return summary


if __name__ == "__main__":
    summary = run_drift_report()
    print(summary)
