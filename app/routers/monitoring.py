from fastapi import APIRouter, HTTPException
from app.schemas import DriftReportResponse
from src.monitoring.drift_detector import run_drift_report

router = APIRouter()


@router.get("/drift-report", response_model=DriftReportResponse)
def get_drift_report():
    """Run feature drift detection and return current status."""
    try:
        summary = run_drift_report()
        return DriftReportResponse(**summary)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {e}")
