"""
Behavioral change point detection using the PELT algorithm (ruptures library).

For each user, fits PELT on their daily total_secs time series and detects
the most significant regime shift. Exports change_points.csv which is joined
into the feature mart and used to anchor LLM explanations.

Run:
    python src/features/change_point_detection.py
"""
import numpy as np
import pandas as pd
import ruptures as rpt
from loguru import logger
from src.utils.config import EXPORTS_DIR
from src.utils.db import query


MIN_DAYS        = 14    # skip users with too few data points
MAX_BKPS        = 1     # we want the single most significant change
PELT_PENALTY    = 10    # higher = fewer breakpoints detected
SAMPLE_SIZE     = None  # set to int to limit users for testing


def fetch_user_logs() -> pd.DataFrame:
    logger.info("Fetching user activity time series from warehouse...")
    sql = """
        SELECT
            msno,
            log_date,
            total_secs
        FROM stg_user_logs
        ORDER BY msno, log_date
    """
    return query(sql)


def detect_change_points(series: np.ndarray) -> int | None:
    """
    Run PELT on a 1D numpy array and return the index of the change point.
    Returns None if no significant change detected.
    """
    if len(series) < MIN_DAYS:
        return None
    try:
        algo = rpt.Pelt(model="rbf").fit(series.reshape(-1, 1))
        result = algo.predict(pen=PELT_PENALTY)
        # result[-1] is always len(series), the true breakpoint is result[-2]
        if len(result) > 1:
            cp_idx = result[-2]
            # Ignore if change point is at the very start or end
            if MIN_DAYS // 2 < cp_idx < len(series) - MIN_DAYS // 2:
                return cp_idx
    except Exception:
        pass
    return None


def compute_post_change_slope(series: np.ndarray, cp_idx: int) -> float:
    """Linear slope of usage after the change point (negative = declining)."""
    post = series[cp_idx:]
    if len(post) < 3:
        return 0.0
    x = np.arange(len(post), dtype=float)
    slope = np.polyfit(x, post, 1)[0]
    return float(slope)


def run(logs: pd.DataFrame) -> pd.DataFrame:
    users = logs.groupby("msno")
    if SAMPLE_SIZE:
        user_list = list(users.groups.keys())[:SAMPLE_SIZE]
    else:
        user_list = list(users.groups.keys())

    logger.info(f"Running PELT on {len(user_list):,} users...")

    records = []
    for i, msno in enumerate(user_list):
        if i % 5000 == 0:
            logger.info(f"  Progress: {i:,} / {len(user_list):,}")

        user_data = users.get_group(msno).sort_values("log_date")
        dates  = user_data["log_date"].values
        values = user_data["total_secs"].fillna(0).values.astype(float)

        cp_idx = detect_change_points(values)

        if cp_idx is not None:
            cp_date  = dates[cp_idx]
            slope    = compute_post_change_slope(values, cp_idx)
            pre_mean = float(values[:cp_idx].mean())
            post_mean = float(values[cp_idx:].mean())
            pct_change = (post_mean - pre_mean) / (pre_mean + 1e-9)
        else:
            cp_date   = None
            slope     = None
            pre_mean  = None
            post_mean = None
            pct_change = None

        records.append({
            "msno":                     msno,
            "change_point_detected":    cp_idx is not None,
            "change_point_date":        cp_date,
            "days_since_change_point":  (pd.Timestamp(dates[-1]) - pd.Timestamp(cp_date)).days if cp_date is not None else None,
            "usage_slope_post_change":  slope,
            "pre_change_daily_mean":    pre_mean,
            "post_change_daily_mean":   post_mean,
            "pct_usage_change":         pct_change,
        })

    return pd.DataFrame(records)


def main() -> None:
    logs = fetch_user_logs()
    change_points = run(logs)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORTS_DIR / "change_points.csv"
    change_points.to_csv(out_path, index=False)

    detected = change_points["change_point_detected"].sum()
    logger.success(
        f"Change point detection complete: "
        f"{detected:,} / {len(change_points):,} users had a detectable regime shift → {out_path}"
    )


if __name__ == "__main__":
    main()
