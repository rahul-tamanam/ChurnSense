"""
DuckDB connection helper.
Provides a context manager and a simple query helper used throughout the project.
"""
import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger
from src.utils.config import WAREHOUSE_PATH


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the project warehouse."""
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(WAREHOUSE_PATH), read_only=read_only)
    return conn


def query(sql: str, params: list = None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    with get_connection(read_only=True) as conn:
        if params:
            return conn.execute(sql, params).df()
        return conn.execute(sql).df()


def execute_sql_file(path: Path, conn: duckdb.DuckDBPyConnection = None) -> None:
    """Read a .sql file and execute it against the warehouse."""
    sql = path.read_text()
    logger.info(f"Executing {path.name}")
    if conn:
        conn.execute(sql)
    else:
        with get_connection() as c:
            c.execute(sql)


def execute_sql_dir(directory: Path, conn: duckdb.DuckDBPyConnection = None) -> None:
    """Execute all .sql files in a directory, sorted by filename."""
    sql_files = sorted(directory.glob("*.sql"))
    if not sql_files:
        logger.warning(f"No .sql files found in {directory}")
        return
    for f in sql_files:
        execute_sql_file(f, conn=conn)
