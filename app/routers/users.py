import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from app.schemas import ChurnRiskResponse
from src.utils.config import EXPORTS_DIR
from src.explainability.shap_explainer import top_factors
from src.explainability.llm_narrator import narrate
from urllib.parse import unquote_plus

router = APIRouter()

_scored_cache: pd.DataFrame | None = None


def get_scored() -> pd.DataFrame:
    global _scored_cache
    if _scored_cache is None:
        path = EXPORTS_DIR / "scored_users.csv"
        if not path.exists():
            raise FileNotFoundError(
                "scored_users.csv not found. Run: python pipelines/scoring_pipeline.py"
            )
        _scored_cache = pd.read_csv(path)
        # Strip rows with null/empty msno that may have slipped through the source data.
        _scored_cache = _scored_cache.dropna(subset=["msno"])
        _scored_cache = _scored_cache[_scored_cache["msno"].astype(str).str.strip() != ""]
        logger.info(f"Loaded scored_users.csv: {len(_scored_cache):,} users")
    return _scored_cache


@router.get("/churn-risk", response_model=ChurnRiskResponse)
def get_churn_risk(
    request: Request,
    msno: str = Query(..., description="The user's MSNO identifier (base64-encoded). Example: PYwMZ/p+5drpbnKCGlCGSzhpomb2/vEpeWwmW58dacQ="),
):
    """
    Return churn probability, SHAP risk factors, and an LLM explanation.

    The `msno` parameter is re-parsed from the raw URL to preserve `+` characters
    in base64-encoded identifiers (FastAPI's default decoding converts `+` to a space).

    Example:
        /users/churn-risk?msno=PYwMZ/p+5drpbnKCGlCGSzhpomb2/vEpeWwmW58dacQ=
    """
    # Re-parse msno from the raw query string so that + chars in base64 MSNOs are
    # kept intact.  FastAPI's standard URL decoding turns + into a space;
    # unquote_plus correctly restores the original + character.
    raw_query = request.url.query
    for part in raw_query.split("&"):
        if part.startswith("msno="):
            parsed = unquote_plus(part[5:])
            if parsed:          # only override if we got something non-empty
                msno = parsed
            break

    # msno is guaranteed non-empty here because FastAPI enforces Query(...) is
    # required, but guard against a caller passing msno= with no value.
    if not msno or not msno.strip():
        raise HTTPException(status_code=422, detail="msno must not be empty")

    try:
        df = get_scored()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    row = df[df["msno"] == msno]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User '{msno}' not found in scored dataset."
        )

    row = row.iloc[0]
    is_high_risk = bool(row["churn_flag"])

    # SHAP values only exist for high-risk users
    shap_cols = {k: v for k, v in row.items() if str(k).startswith("shap_")}
    has_shap  = any(pd.notna(v) for v in shap_cols.values())
    risk_factors = top_factors(pd.Series(shap_cols)) if has_shap else []

    # Change-point data (optional)
    change_point: dict = {"change_point_detected": False}
    cp_path = EXPORTS_DIR / "change_points.csv"
    if cp_path.exists():
        cp_df  = pd.read_csv(cp_path)
        cp_row = cp_df[cp_df["msno"] == msno]
        if not cp_row.empty:
            change_point = cp_row.iloc[0].to_dict()

    # Call LLM narrator only for high-risk users that have SHAP explanations
    if is_high_risk and has_shap:
        user_data = row.to_dict()
        narrative = narrate(user_data, shap_cols, change_point)
    else:
        narrative = {
            "explanation":          "This user is not currently flagged as high-risk.",
            "recommended_action":   "no_action",
            "action_justification": "Churn probability is below the risk threshold.",
        }

    cp_date_raw = change_point.get("change_point_date")
    cp_date_str = (
        str(cp_date_raw)
        if cp_date_raw and str(cp_date_raw) not in ("", "nan", "None")
        else None
    )

    return ChurnRiskResponse(
        msno=msno,
        churn_probability=float(row["churn_prob"]),
        churn_flag=is_high_risk,
        explanation=narrative["explanation"],
        recommended_action=narrative["recommended_action"],
        action_justification=narrative["action_justification"],
        top_risk_factors=risk_factors,
        uplift_score=float(row["uplift_score"]) if pd.notna(row.get("uplift_score")) else None,
        change_point_detected=bool(change_point.get("change_point_detected", False)),
        change_point_date=cp_date_str,
    )