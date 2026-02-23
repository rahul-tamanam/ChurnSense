import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schemas import ChurnRiskResponse
from src.utils.config import EXPORTS_DIR

router = APIRouter()


def _load_action_plan() -> pd.DataFrame:
    path = EXPORTS_DIR / "action_plan.csv"
    if not path.exists():
        raise FileNotFoundError("action_plan.csv not found. Run the pipeline first.")
    return pd.read_csv(path)


@router.get("/{msno}/churn-risk", response_model=ChurnRiskResponse)
def get_churn_risk(msno: str):
    """Return churn probability and explanation for a user."""
    try:
        df = _load_action_plan()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    row = df[df["msno"] == msno]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User {msno} not found or not high-risk.")

    row = row.iloc[0]
    return ChurnRiskResponse(
        msno=msno,
        churn_probability=float(row.get("churn_prob", 0)),
        churn_flag=True,
        explanation=str(row.get("explanation", "")),
        top_risk_factors=[],   # populated by SHAP in full implementation
        change_point_detected=False,
        change_point_date=None,
    )
