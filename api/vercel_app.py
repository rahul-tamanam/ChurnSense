"""
Slim FastAPI app for Vercel serverless (portfolio demo).
Uses only public/data/*.csv — no XGBoost, SHAP lib, Groq, or Evidently.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas import ChurnRiskResponse, DriftReportResponse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "public" / "data"

DRIFT_DROP = [
    "msno", "is_churn", "split", "behavioral_cohort", "risk_tier",
    "change_point_date", "last_intervention_type",
]


def _manual_psi_drift(ref: pd.DataFrame, cur: pd.DataFrame, threshold: float = 0.2):
    drifted = 0
    total = len(ref.columns)
    for col in ref.columns:
        try:
            ref_vals = ref[col].dropna()
            cur_vals = cur[col].dropna()
            bins = np.histogram_bin_edges(ref_vals, bins=10)
            ref_pct = np.histogram(ref_vals, bins=bins)[0] / len(ref_vals)
            cur_pct = np.histogram(cur_vals, bins=bins)[0] / max(len(cur_vals), 1)
            ref_pct = np.clip(ref_pct, 1e-6, None)
            cur_pct = np.clip(cur_pct, 1e-6, None)
            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            if psi > threshold:
                drifted += 1
        except Exception:
            continue
    share = drifted / total if total else 0.0
    return drifted, total, share


def top_factors(shap_row: pd.Series, n: int = 5) -> list[dict]:
    items = [
        (col.replace("shap_", ""), val)
        for col, val in shap_row.items()
        if col.startswith("shap_")
    ]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "direction": "risk_increase" if val > 0 else "risk_decrease",
        }
        for feat, val in items[:n]
    ]


def demo_narrative(
    churn_prob: float,
    factors: list[dict],
    change_point: dict,
) -> dict:
    feat = factors[0]["feature"].replace("_", " ") if factors else "usage patterns"
    cp_line = ""
    if change_point.get("change_point_detected"):
        d = change_point.get("change_point_date") or "a recent date"
        cp_line = f" A behavioral shift was flagged around {d}."
    return {
        "explanation": (
            f"(Portfolio demo — static narrative.) Risk is driven mainly by {feat}. "
            f"Estimated churn probability is {churn_prob:.0%}.{cp_line}"
        ),
        "recommended_action": "personal_outreach",
        "action_justification": (
            "Demo mode recommends outreach; wire GROQ_API_KEY on your own host for live LLM text."
        ),
    }


_scored_cache: pd.DataFrame | None = None


def get_scored() -> pd.DataFrame:
    global _scored_cache
    if _scored_cache is None:
        path = DATA_DIR / "scored_users.csv"
        if not path.exists():
            raise FileNotFoundError(f"Demo data missing: {path}")
        _scored_cache = pd.read_csv(path)
        _scored_cache = _scored_cache.dropna(subset=["msno"])
        _scored_cache = _scored_cache[_scored_cache["msno"].astype(str).str.strip() != ""]
    return _scored_cache


app = FastAPI(
    title="Churn Prevention System (demo)",
    description="Portfolio demo — sample CSVs in public/data.",
    version="demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/users/churn-risk", response_model=ChurnRiskResponse)
def get_churn_risk(
    request: Request,
    msno: str = Query(..., description="User identifier (same as scored_users.csv)."),
):
    raw_query = request.url.query
    for part in raw_query.split("&"):
        if part.startswith("msno="):
            parsed = unquote_plus(part[5:])
            if parsed:
                msno = parsed
            break

    if not msno or not msno.strip():
        raise HTTPException(status_code=422, detail="msno must not be empty")

    try:
        df = get_scored()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    row = df[df["msno"] == msno]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User '{msno}' not found in demo dataset.")

    row = row.iloc[0]
    is_high_risk = bool(row["churn_flag"])

    shap_cols = {k: v for k, v in row.items() if str(k).startswith("shap_")}
    has_shap = any(pd.notna(v) for v in shap_cols.values())
    risk_factors = top_factors(pd.Series(shap_cols)) if has_shap else []

    change_point: dict = {"change_point_detected": False}
    cp_path = DATA_DIR / "change_points.csv"
    if cp_path.exists():
        cp_df = pd.read_csv(cp_path)
        cp_row = cp_df[cp_df["msno"] == msno]
        if not cp_row.empty:
            change_point = cp_row.iloc[0].to_dict()

    if is_high_risk and has_shap:
        narrative = demo_narrative(float(row["churn_prob"]), risk_factors, change_point)
    else:
        narrative = {
            "explanation": "This user is not flagged as high-risk in the demo slice.",
            "recommended_action": "no_action",
            "action_justification": "Churn probability is below the demo threshold.",
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


@app.get("/monitoring/drift-report", response_model=DriftReportResponse)
def get_drift_report():
    ref_path = DATA_DIR / "features_train.csv"
    cur_path = DATA_DIR / "features_score.csv"
    if not ref_path.exists() or not cur_path.exists():
        # Friendly demo defaults if optional drift files absent
        return DriftReportResponse(
            dataset_drift_detected=False,
            drifted_features=0,
            total_features=0,
            drift_share=0.0,
            retraining_recommended=False,
        )

    reference = pd.read_csv(ref_path)
    current = pd.read_csv(cur_path)

    ref_feat = reference.drop(columns=[c for c in DRIFT_DROP if c in reference.columns])
    cur_feat = current.drop(columns=[c for c in DRIFT_DROP if c in current.columns])

    common = [c for c in ref_feat.columns if c in cur_feat.columns]
    ref_feat = ref_feat[common].fillna(0)
    cur_feat = cur_feat[common].fillna(0)

    drifted, total, share = _manual_psi_drift(ref_feat, cur_feat)
    threshold = 0.2
    return DriftReportResponse(
        dataset_drift_detected=share > threshold,
        drifted_features=int(drifted),
        total_features=int(total),
        drift_share=round(float(share), 4),
        retraining_recommended=share > threshold,
    )
