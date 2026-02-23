"""
Intervention Selector
----------------------
Given a user's churn score, uplift scores per treatment type, and LLM recommendation,
picks the optimal intervention.

Logic:
  1. If uplift score < UPLIFT_MIN_SCORE → no_action (user is not persuadable)
  2. Otherwise, pick the intervention with highest uplift that aligns with LLM recommendation
  3. Output a ranked action plan per user
"""
import pandas as pd
from loguru import logger
from src.utils.config import UPLIFT_MIN_SCORE
from src.models.uplift_model import compute_uplift
from src.models.model_registry import load_metadata


INTERVENTION_TYPES = [
    "discount_offer",
    "feature_highlight_email",
    "personal_outreach",
    "re_onboarding_flow",
]


def select_interventions(
    scored_df: pd.DataFrame,
    features_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine churn scores + uplift + LLM recommendations into final action plan.

    Returns DataFrame with one row per high-risk user with:
      - churn_prob
      - uplift_score
      - recommended_action (from LLM)
      - final_action (LLM recommendation validated by uplift)
      - explanation
      - action_justification
    """
    high_risk = scored_df[scored_df["churn_flag"] == 1].copy()

    # Merge features for uplift computation
    meta = load_metadata("uplift_treatment")
    available_features = meta.get("features", [])
    feat_subset = features_df[[c for c in available_features if c in features_df.columns]]

    uplift_scores = compute_uplift(feat_subset, available_features)
    high_risk = high_risk.merge(
        pd.DataFrame({"msno": features_df["msno"], "uplift_score": uplift_scores.values}),
        on="msno", how="left"
    )

    # Merge LLM narratives
    high_risk = high_risk.merge(narratives_df, on="msno", how="left")

    # Determine final action
    def resolve_action(row):
        if pd.isna(row.get("uplift_score")) or row["uplift_score"] < UPLIFT_MIN_SCORE:
            return "no_action"
        llm_action = row.get("recommended_action", "no_action")
        return llm_action if llm_action in INTERVENTION_TYPES else "no_action"

    high_risk["final_action"] = high_risk.apply(resolve_action, axis=1)

    actionable = high_risk[high_risk["final_action"] != "no_action"]
    logger.info(
        f"Intervention plan: {len(actionable):,} users get intervention | "
        f"{(high_risk['final_action']=='no_action').sum():,} not persuadable"
    )

    output_cols = [
        "msno", "churn_prob", "uplift_score",
        "final_action", "explanation", "action_justification"
    ]
    return high_risk[[c for c in output_cols if c in high_risk.columns]]
