from pydantic import BaseModel
from typing import Optional


class ChurnRiskResponse(BaseModel):
    msno:                  str
    churn_probability:     float
    churn_flag:            bool
    explanation:           str
    recommended_action:    str
    action_justification:  str
    top_risk_factors:      list[dict]
    uplift_score:          Optional[float]
    change_point_detected: bool
    change_point_date:     Optional[str]


class DriftReportResponse(BaseModel):
    dataset_drift_detected:  bool
    drifted_features:        int
    total_features:          int
    drift_share:             float
    retraining_recommended:  bool