from pydantic import BaseModel
from typing import Optional


class ChurnRiskResponse(BaseModel):
    msno:              str
    churn_probability: float
    churn_flag:        bool
    explanation:       str
    top_risk_factors:  list[dict]
    change_point_detected: bool
    change_point_date: Optional[str]


class InterventionResponse(BaseModel):
    msno:                 str
    churn_probability:    float
    uplift_score:         Optional[float]
    final_action:         str
    explanation:          str
    action_justification: str


class DriftReportResponse(BaseModel):
    dataset_drift_detected:  bool
    drifted_features:        int
    total_features:          int
    drift_share:             float
    retraining_recommended:  bool
