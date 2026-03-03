from pathlib import Path
import sys

from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `import app` works when tests run in isolation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"status": "ok"}


def test_drift_report_503_on_file_not_found(monkeypatch):
    """
    If the underlying drift detector raises FileNotFoundError, the API should
    surface a 503 with a clear message.
    """
    # Patch the function used by the monitoring router to simulate missing files.
    from app.routers import monitoring as monitoring_router

    def fake_run_drift_report(*args, **kwargs):
        raise FileNotFoundError("features_train.csv not found")

    monkeypatch.setattr(
        monitoring_router,
        "run_drift_report",
        fake_run_drift_report,
        raising=True,
    )

    resp = client.get("/monitoring/drift-report")
    assert resp.status_code == 503
    body = resp.json()
    assert "detail" in body
    assert "features_train.csv not found" in body["detail"]


def test_churn_risk_missing_msno_returns_422():
    resp = client.get("/users/churn-risk")  # no msno query param
    assert resp.status_code == 422

