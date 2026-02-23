"""
Central config — loads .env and exposes typed constants used across the project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).resolve().parents[2]
KKBOX_RAW_DIR     = ROOT / os.getenv("KKBOX_RAW_DIR",     "data/raw/kkbox")
SYNTHETIC_RAW_DIR = ROOT / os.getenv("SYNTHETIC_RAW_DIR", "data/raw/synthetic")
WAREHOUSE_PATH    = ROOT / os.getenv("WAREHOUSE_PATH",     "data/warehouse/churn.duckdb")
EXPORTS_DIR       = ROOT / os.getenv("EXPORTS_DIR",        "data/exports")
MODELS_DIR        = ROOT / os.getenv("MODELS_DIR",         "models")
SQL_DIR           = ROOT / "sql"

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Model config ──────────────────────────────────────────────────────────────
CHURN_THRESHOLD   = float(os.getenv("CHURN_THRESHOLD",  "0.65"))
UPLIFT_MIN_SCORE  = float(os.getenv("UPLIFT_MIN_SCORE", "0.05"))

# ── Monitoring ────────────────────────────────────────────────────────────────
DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))

# ── Feature columns used by ML models ─────────────────────────────────────────
CHURN_FEATURES = [
    # Usage decay
    "usage_decay_30d", "usage_decay_60d", "usage_decay_90d",
    "sessions_last_30d", "sessions_last_60d", "sessions_last_90d",
    "avg_session_duration_30d",
    # Login frequency
    "login_days_last_30d", "login_gap_days_avg", "login_streak_max",
    # Support signals
    "ticket_count_30d", "ticket_count_90d",
    "pct_negative_sentiment", "unresolved_ticket_count",
    "has_cancellation_inquiry",
    # Transaction signals
    "plan_downgrade_flag", "payment_failure_count_90d",
    "auto_renew_flag", "days_since_last_txn",
    # Feature adoption
    "unique_features_used_30d", "feature_diversity_score",
    "days_since_last_feature_use",
    # Behavioral change
    "change_point_detected", "days_since_change_point",
    "usage_slope_post_change",
    # Demographics
    "tenure_days", "city_encoded", "age_bucket",
]

UPLIFT_FEATURES = CHURN_FEATURES + ["intervention_type_encoded"]
