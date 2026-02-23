"""
FastAPI serving layer for the churn prevention system.

Endpoints:
    GET  /users/{msno}/churn-risk       → churn probability + explanation
    GET  /users/{msno}/intervention     → recommended action + message
    GET  /monitoring/drift-report       → current feature drift status

Run:
    uvicorn app.main:app --reload --port 8000
    # or
    make api
"""
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.routers import users, interventions, monitoring

app = FastAPI(
    title="Churn Prevention System",
    description="Real-time churn risk scoring, explanation, and intervention recommendations.",
    version="1.0.0",
)

app.include_router(users.router,         prefix="/users",      tags=["Churn Risk"])
app.include_router(interventions.router, prefix="/users",      tags=["Interventions"])
app.include_router(monitoring.router,    prefix="/monitoring", tags=["Monitoring"])


@app.get("/health")
def health():
    return {"status": "ok"}
