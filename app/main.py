from fastapi import FastAPI
from app.routers import users, monitoring, interventions

app = FastAPI(
    title="Churn Prevention System",
    description="Real-time churn risk scoring, explanation, and intervention recommendations.",
    version="1.0.0",
)

app.include_router(users.router,         prefix="/users",         tags=["Churn Risk"])
app.include_router(monitoring.router,    prefix="/monitoring",    tags=["Monitoring"])
app.include_router(interventions.router, prefix="/interventions", tags=["Interventions"])

@app.get("/health")
def health():
    return {"status": "ok"}