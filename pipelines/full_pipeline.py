"""
Full Pipeline
-------------
End-to-end: ingest → features → train → score → explain → interventions

Run:
    python pipelines/full_pipeline.py
    # or
    make pipeline
"""
from loguru import logger

from src.ingestion.load_to_warehouse import main as ingest
from src.features.run_feature_pipeline import main as run_features
from src.features.change_point_detection import main as detect_change_points
from src.models.churn_model import main as train_churn, score as score_churn
from src.models.uplift_model import main as train_uplift
from src.explainability.llm_narrator import narrate_batch
from src.interventions.intervention_selector import select_interventions
from src.utils.config import EXPORTS_DIR

import pandas as pd


def main():
    logger.info("=" * 60)
    logger.info("CHURN PREVENTION PIPELINE — FULL RUN")
    logger.info("=" * 60)

    # Step 1: Ingest
    logger.info("[1/7] Ingesting data into DuckDB warehouse...")
    ingest()

    # Step 2: Feature engineering
    logger.info("[2/7] Running SQL feature pipeline...")
    run_features()

    # Step 3: Change point detection
    logger.info("[3/7] Detecting behavioral change points...")
    detect_change_points()

    # Step 4: Train models
    logger.info("[4/7] Training churn classifier...")
    train_churn(do_train=True)
    logger.info("[4/7] Training uplift models...")
    train_uplift(do_train=True)

    # Step 5: Score users
    logger.info("[5/7] Scoring users for churn risk...")
    features_df   = pd.read_csv(EXPORTS_DIR / "features_score.csv")
    scored_df     = score_churn(features_df)

    # Step 6: Generate LLM narratives
    logger.info("[6/7] Generating LLM explanations and recommendations...")
    change_pts_df = pd.read_csv(EXPORTS_DIR / "change_points.csv")
    narratives_df = narrate_batch(scored_df, change_pts_df, features_df)

    # Step 7: Build intervention plan
    logger.info("[7/7] Selecting interventions...")
    action_plan   = select_interventions(scored_df, features_df, narratives_df)

    out_path = EXPORTS_DIR / "action_plan.csv"
    action_plan.to_csv(out_path, index=False)

    high_risk_count = len(action_plan)
    actioned        = (action_plan["final_action"] != "no_action").sum()
    logger.success(
        f"Pipeline complete. "
        f"{high_risk_count:,} high-risk users identified. "
        f"{actioned:,} will receive intervention. "
        f"Action plan → {out_path}"
    )


if __name__ == "__main__":
    main()
