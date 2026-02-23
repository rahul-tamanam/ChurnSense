import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schemas import InterventionResponse
from src.utils.config import EXPORTS_DIR

router = APIRouter()


@router.get("/{msno}/intervention", response_model=InterventionResponse)
def get_intervention(msno: str):
    """Return recommended retention intervention for a high-risk user."""
    path = EXPORTS_DIR / "action_plan.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="action_plan.csv not found. Run the pipeline first.")

    df  = pd.read_csv(path)
    row = df[df["msno"] == msno]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User {msno} not found or not high-risk.")

    row = row.iloc[0]
    return InterventionResponse(
        msno=msno,
        churn_probability=float(row.get("churn_prob", 0)),
        uplift_score=float(row["uplift_score"]) if pd.notna(row.get("uplift_score")) else None,
        final_action=str(row.get("final_action", "no_action")),
        explanation=str(row.get("explanation", "")),
        action_justification=str(row.get("action_justification", "")),
    )
