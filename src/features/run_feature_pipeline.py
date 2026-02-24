from pathlib import Path
from loguru import logger
from src.utils.config import SQL_DIR, EXPORTS_DIR
from src.utils.db import execute_sql_file, query, get_connection


# Explicit execution order — dependencies must run before dependents
FEATURE_SQL_ORDER = [
    SQL_DIR / "03_features" / "feat_sessionization.sql",
    SQL_DIR / "03_features" / "feat_login_frequency.sql",
    SQL_DIR / "03_features" / "feat_usage_decay.sql",
    SQL_DIR / "03_features" / "feat_support_signals.sql",
    SQL_DIR / "03_features" / "feat_transaction_signals.sql",
    SQL_DIR / "03_features" / "feat_master.sql",
]

MART_SQL_ORDER = [
    SQL_DIR / "04_marts" / "mart_churn_features.sql",
    SQL_DIR / "04_marts" / "mart_intervention_results.sql",
]


def load_change_points_if_ready(conn) -> None:
    cp_path = EXPORTS_DIR / "change_points.csv"
    if cp_path.exists():
        logger.info("Loading change_points.csv into warehouse...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE raw_change_points AS
            SELECT * FROM read_csv_auto('{cp_path}', ignore_errors=true)
        """)
        count = conn.execute("SELECT COUNT(*) FROM raw_change_points").fetchone()[0]
        logger.info(f"  Loaded {count:,} change point records")
    else:
        logger.info(
            "change_points.csv not found — feat_master will use empty fallback. "
            "Run change_point_detection.py then re-run to include PELT features."
        )


def run_feature_sql(conn) -> None:
    logger.info("Running feature SQL in dependency order...")
    for path in FEATURE_SQL_ORDER:
        execute_sql_file(path, conn=conn)

    logger.info("Running mart SQL...")
    for path in MART_SQL_ORDER:
        execute_sql_file(path, conn=conn)


def export_feature_mart() -> None:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    exports = {
        "features_train.csv":       "SELECT * FROM mart_churn_features WHERE split = 'train'",
        "features_score.csv":       "SELECT * FROM mart_churn_features",
        "intervention_results.csv": "SELECT * FROM mart_intervention_results",
    }
    for filename, sql in exports.items():
        try:
            df = query(sql)
            path = EXPORTS_DIR / filename
            df.to_csv(path, index=False)
            logger.info(f"  Exported {filename}: {len(df):,} rows → {path}")
        except Exception as e:
            logger.warning(f"  Could not export {filename}: {e}")


def main() -> None:
    logger.info("Starting feature pipeline")
    with get_connection() as conn:
        load_change_points_if_ready(conn)
        run_feature_sql(conn)
    export_feature_mart()
    logger.success("Feature pipeline complete → data/exports/")


if __name__ == "__main__":
    main()