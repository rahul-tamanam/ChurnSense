"""
FastAPI app — serves the React dashboard at / and the API at /users + /monitoring.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routers import users, monitoring


app = FastAPI(
    title="Churn Prevention System",
    description="Real-time churn risk scoring, explanation, and intervention recommendations.",
    version="1.0.0",
)


# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(users.router,      prefix="/users",      tags=["Churn Risk"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])


# ── Serve CSV exports so the dashboard can fetch them ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
if EXPORTS_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(EXPORTS_DIR)), name="data")


# ── Serve React dashboard (static index.html under app/dashboard) ────────────
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


@app.get("/", include_in_schema=False)
def serve_dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Dashboard not found. Make sure app/dashboard/index.html exists."}


@app.get("/health")
def health():
    return {"status": "ok"}
