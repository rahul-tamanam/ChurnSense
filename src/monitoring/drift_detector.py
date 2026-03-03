"""
Feature Drift Detector
-----------------------
Compares reference (training) vs current (scoring) feature distributions.
Flags drift and triggers retraining recommendation if threshold exceeded.
"""
from pathlib import Path

import pandas as pd
from loguru import logger
from src.utils.config import EXPORTS_DIR, DRIFT_PSI_THRESHOLD


def run_drift_report(
    reference_path: str = None,
    current_path:   str = None,
) -> dict:
    ref_path = Path(reference_path) if reference_path else EXPORTS_DIR / "features_train.csv"
    cur_path = Path(current_path)   if current_path   else EXPORTS_DIR / "features_score.csv"

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference feature file not found at {ref_path}")
    if not cur_path.exists():
        raise FileNotFoundError(f"Current feature file not found at {cur_path}")

    reference = pd.read_csv(ref_path)
    current   = pd.read_csv(cur_path)

    drop_cols = ["msno", "is_churn", "split", "behavioral_cohort", "risk_tier",
                 "change_point_date", "last_intervention_type"]
    ref_feat  = reference.drop(columns=[c for c in drop_cols if c in reference.columns])
    cur_feat  = current.drop(columns=[c for c in drop_cols if c in current.columns])

    common    = [c for c in ref_feat.columns if c in cur_feat.columns]
    ref_feat  = ref_feat[common].fillna(0)
    cur_feat  = cur_feat[common].fillna(0)

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_feat, current_data=cur_feat)
        result = report.as_dict()
        metrics = result["metrics"][0]["result"]
        drifted  = metrics.get("number_of_drifted_columns", 0)
        total    = metrics.get("number_of_columns", len(common))
        share    = metrics.get("share_of_drifted_columns", 0)

    except Exception:
        # Fallback: manual PSI-based drift detection
        logger.info("Evidently unavailable — using manual PSI drift detection")
        drifted, total, share = _manual_psi_drift(ref_feat, cur_feat)

    summary = {
        "dataset_drift_detected": share > DRIFT_PSI_THRESHOLD,
        "drifted_features":       int(drifted),
        "total_features":         int(total),
        "drift_share":            round(float(share), 4),
        "retraining_recommended": share > DRIFT_PSI_THRESHOLD,
    }

    logger.info(
        f"Drift report: {drifted}/{total} features drifted "
        f"({share:.0%}) | retraining={summary['retraining_recommended']}"
    )
    return summary


def _manual_psi_drift(ref: pd.DataFrame, cur: pd.DataFrame, threshold: float = 0.2):
    """Simple PSI-based drift detection as fallback."""
    import numpy as np
    drifted = 0
    total   = len(ref.columns)

    for col in ref.columns:
        try:
            ref_vals = ref[col].dropna()
            cur_vals = cur[col].dropna()
            bins     = np.histogram_bin_edges(ref_vals, bins=10)
            ref_pct  = np.histogram(ref_vals, bins=bins)[0] / len(ref_vals)
            cur_pct  = np.histogram(cur_vals, bins=bins)[0] / max(len(cur_vals), 1)
            ref_pct  = np.clip(ref_pct, 1e-6, None)
            cur_pct  = np.clip(cur_pct, 1e-6, None)
            psi      = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            if psi > threshold:
                drifted += 1
        except Exception:
            continue

    share = drifted / total if total else 0
    return drifted, total, share


if __name__ == "__main__":
    summary = run_drift_report()
    print(summary)