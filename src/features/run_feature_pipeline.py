"""
Orchestrates all SQL feature engineering in order, then exports the
final feature mart to data/exports/ for ML consumption.

Run:
    python src/features/run_feature_pipeline.py
    # or
    make features
"""
from loguru import logger
from src.utils.config import SQL_DIR, EXPORTS_DIR
from src.utils.db import execute_sql_dir, execute_sql_file, query, get_connection


def run_feature_sql(conn) -> None:
    logger.info("Running feature SQL transforms...")
    execute_sql_dir(SQL_DIR / "03_features", conn=conn)
    logger.info("Running feature marts...")
    execute_sql_dir(SQL_DIR / "04_marts", conn=conn)


def export_feature_mart() -> None:
    """Export final feature tables to CSV for ML training."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    exports = {
        "features_train.csv": "SELECT * FROM mart_churn_features WHERE split = 'train'",
        "features_score.csv": "SELECT * FROM mart_churn_features WHERE split = 'score'",
        "intervention_results.csv": "SELECT * FROM mart_intervention_results",
    }
    for filename, sql in exports.items():
        df = query(sql)
        path = EXPORTS_DIR / filename
        df.to_csv(path, index=False)
        logger.info(f"  Exported {filename}: {len(df):,} rows → {path}")


def main() -> None:
    logger.info("Starting feature pipeline")

    with get_connection() as conn:
        run_feature_sql(conn)

    export_feature_mart()
    logger.success("Feature pipeline complete → data/exports/")


if __name__ == "__main__":
    main()
