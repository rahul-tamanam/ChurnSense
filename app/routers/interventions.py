from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd

from src.utils.config import EXPORTS_DIR


router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def action_plan_dashboard() -> str:
    """
    Simple HTML dashboard for recruiters:
    - Reads action_plan.csv from data/exports/
    - Shows a summary and top-N high-risk users with planned interventions.
    """
    path = EXPORTS_DIR / "action_plan.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Action plan not found at {path}. Run: make pipeline",
        )

    df = pd.read_csv(path)
    total_users = len(df)
    n_actionable = (df["final_action"] != "no_action").sum()
    top = df.sort_values("churn_prob", ascending=False).head(200)

    # ── Segment insights: join with features_score for risk_tier / behavioral_cohort ──
    segment_html = ""
    try:
        features_path = EXPORTS_DIR / "features_score.csv"
        if features_path.exists():
            feat = pd.read_csv(
                features_path,
                usecols=["msno", "behavioral_cohort", "risk_tier"],
            )
            joined = df.merge(feat, on="msno", how="left")

            # Risk tier summary
            if "risk_tier" in joined.columns:
                risk_summary = (
                    joined.groupby("risk_tier")
                    .agg(
                        high_risk_users=("msno", "nunique"),
                        avg_churn_prob=("churn_prob", "mean"),
                        intervention_rate=(
                            "final_action",
                            lambda s: (s != "no_action").mean() * 100,
                        ),
                    )
                    .reset_index()
                    .sort_values("avg_churn_prob", ascending=False)
                )
                risk_summary["intervention_rate"] = risk_summary[
                    "intervention_rate"
                ].round(1)
                risk_table = risk_summary.to_html(
                    classes="table table-sm table-striped",
                    index=False,
                    float_format=lambda x: f"{x:0.3f}"
                    if isinstance(x, float)
                    else x,
                )
            else:
                risk_table = ""

            # Behavioral cohort summary
            if "behavioral_cohort" in joined.columns:
                cohort_summary = (
                    joined.groupby("behavioral_cohort")
                    .agg(
                        high_risk_users=("msno", "nunique"),
                        avg_churn_prob=("churn_prob", "mean"),
                        intervention_rate=(
                            "final_action",
                            lambda s: (s != "no_action").mean() * 100,
                        ),
                    )
                    .reset_index()
                    .sort_values("avg_churn_prob", ascending=False)
                )
                cohort_summary["intervention_rate"] = cohort_summary[
                    "intervention_rate"
                ].round(1)
                cohort_table = cohort_summary.to_html(
                    classes="table table-sm table-striped",
                    index=False,
                    float_format=lambda x: f"{x:0.3f}"
                    if isinstance(x, float)
                    else x,
                )
            else:
                cohort_table = ""

            if risk_table or cohort_table:
                segment_html = f"""
                <div class="row g-3 mt-4">
                    <div class="col-md-6">
                        <div class="table-container">
                            <h2 class="h6 mb-3">By risk tier</h2>
                            {risk_table}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="table-container">
                            <h2 class="h6 mb-3">By behavioral cohort</h2>
                            {cohort_table}
                        </div>
                    </div>
                </div>
                """
    except Exception:
        # If anything goes wrong, we still want the main dashboard to render.
        segment_html = ""

    # Build a minimal, clean HTML table for individual users
    table_html = top.to_html(
        classes="table table-striped",
        index=False,
        float_format=lambda x: f"{x:0.3f}" if isinstance(x, float) else x,
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Churn Action Plan Dashboard</title>
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" />
        <style>
            body {{
                padding: 2rem;
                background-color: #0f172a;
                color: #e5e7eb;
            }}
            h1, h2 {{
                color: #f9fafb;
            }}
            .metric-card {{
                background: #020617;
                border-radius: 0.75rem;
                padding: 1rem 1.25rem;
                border: 1px solid #1f2937;
            }}
            .table-container {{
                margin-top: 1.5rem;
                background: #020617;
                border-radius: 0.75rem;
                padding: 1rem;
                border: 1px solid #1f2937;
                overflow-x: auto;
            }}
            table {{
                color: #e5e7eb !important;
            }}
            thead th {{
                background-color: #020617 !important;
            }}
            tbody tr:nth-child(even) {{
                background-color: #020617 !important;
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="mb-3">Churn Action Plan</h1>
            <p class="text-muted mb-4">
                Snapshot of high-risk users and their recommended interventions,
                generated by the full pipeline.
            </p>

            <div class="row g-3 mb-3">
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="text-sm text-muted">High-risk users</div>
                        <div class="h4 mb-0">{total_users:,}</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="text-sm text-muted">Users with intervention</div>
                        <div class="h4 mb-0">{n_actionable:,}</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="text-sm text-muted">Intervention rate</div>
                        <div class="h4 mb-0">
                            {(n_actionable / total_users * 100):.1f}%
                        </div>
                    </div>
                </div>
            </div>

            <div class="table-container">
                <h2 class="h5 mb-3">Top {len(top):,} high-risk users</h2>
                {table_html}
            </div>

            {segment_html}
        </div>
    </body>
    </html>
    """
    return html
