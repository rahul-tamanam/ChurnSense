"""
Loads all raw CSVs (KKBox + synthetic) into DuckDB and runs staging transforms.

Run:
    python src/ingestion/load_to_warehouse.py
    # or
    make ingest
"""
import duckdb
from loguru import logger
from src.utils.config import (
    KKBOX_RAW_DIR, SYNTHETIC_RAW_DIR, WAREHOUSE_PATH, SQL_DIR
)
from src.utils.db import execute_sql_dir, get_connection


KKBOX_TABLES = {
    "raw_members":      "members_v3.csv",
    "raw_train":        "train_v2.csv",
    "raw_transactions": "transactions_v2.csv",
    "raw_user_logs":    "user_logs_v2.csv",
}

SYNTHETIC_TABLES = {
    "raw_feature_events":  "feature_events.csv",
    "raw_support_tickets": "support_tickets.csv",
    "raw_interventions":   "interventions.csv",
    "raw_outcomes":        "outcomes.csv",
}


def load_csv_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Register all raw CSVs as DuckDB tables using read_csv_auto."""
    for table, filename in KKBOX_TABLES.items():
        path = KKBOX_RAW_DIR / filename
        if not path.exists():
            logger.warning(f"Missing KKBox file: {path} — skipping")
            continue
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv_auto('{path}', ignore_errors=true)
        """)
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  Loaded {table}: {count:,} rows")

    for table, filename in SYNTHETIC_TABLES.items():
        path = SYNTHETIC_RAW_DIR / filename
        if not path.exists():
            logger.warning(f"Missing synthetic file: {path} — skipping")
            continue
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv_auto('{path}', ignore_errors=true)
        """)
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  Loaded {table}: {count:,} rows")


def run_staging(conn: duckdb.DuckDBPyConnection) -> None:
    """Run all staging SQL transforms."""
    execute_sql_dir(SQL_DIR / "02_staging", conn=conn)


def main() -> None:
    logger.info("Starting ingestion pipeline")
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        logger.info("Loading raw CSVs into DuckDB...")
        load_csv_tables(conn)

        logger.info("Running staging transforms...")
        run_staging(conn)

    logger.success("Ingestion complete → data/warehouse/churn.duckdb")


if __name__ == "__main__":
    main()
