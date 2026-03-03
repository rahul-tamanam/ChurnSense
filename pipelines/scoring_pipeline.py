"""
Scoring Pipeline
----------------
1. Load features_score.csv
2. Score ALL users with churn model → churn_prob + churn_flag
3. Compute SHAP for high-risk users only
4. Compute uplift scores for high-risk users only
5. Save scored_users.csv  ← API reads from this

Narration happens on-demand in the API, not here.

Run:
    python pipelines/scoring_pipeline.py
"""
import pandas as pd
from loguru import logger

from src.utils.config import EXPORTS_DIR, CHURN_FEATURES, CHURN_THRESHOLD
from src.models.model_registry import load_model, load_metadata
from src.explainability.shap_explainer import explain
from src.models.uplift_model import compute_uplift


def load_features() -> pd.DataFrame:
    features_path = EXPORTS_DIR / "features_score.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found. Run the feature pipeline first: make features"
        )
    features_df = pd.read_csv(features_path)

    # Drop rows whose msno is missing or blank — they can't be scored or looked up.
    before = len(features_df)
    features_df = features_df.dropna(subset=["msno"])
    features_df = features_df[features_df["msno"].astype(str).str.strip() != ""]
    dropped = before - len(features_df)
    if dropped:
        logger.warning(f"Dropped {dropped:,} rows with empty/null msno from features_score.csv")

    logger.info(f"Loaded {len(features_df):,} users from features_score.csv")
    return features_df


def score_all_users(features_df: pd.DataFrame) -> pd.DataFrame:
    model     = load_model("churn_classifier")
    available = [c for c in CHURN_FEATURES if c in features_df.columns]
    missing   = set(CHURN_FEATURES) - set(available)
    if missing:
        logger.warning(f"Missing features (filling with 0): {missing}")

    X     = features_df[available].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    scored = features_df[["msno"]].copy()
    scored["churn_prob"] = probs
    scored["churn_flag"] = (probs >= CHURN_THRESHOLD).astype(int)

    n_high = int(scored["churn_flag"].sum())
    logger.info(
        f"Scored {len(scored):,} users | "
        f"High-risk: {n_high:,} ({n_high / len(scored):.1%})"
    )
    return scored


def add_shap(scored_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    high_risk_msno = scored_df.loc[scored_df["churn_flag"] == 1, "msno"]
    high_risk_feat = (
        features_df[features_df["msno"].isin(high_risk_msno)]
        .set_index("msno")
    )

    shap_df = explain(high_risk_feat)
    shap_df = shap_df.reset_index().rename(columns={"index": "msno"})

    scored_df = scored_df.merge(shap_df, on="msno", how="left")
    logger.info(f"SHAP computed for {len(shap_df):,} high-risk users")
    return scored_df


def add_uplift(scored_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    try:
        meta      = load_metadata("uplift_treatment")
        available = [c for c in meta.get("features", CHURN_FEATURES) if c in features_df.columns]

        high_risk_msno = scored_df.loc[scored_df["churn_flag"] == 1, "msno"]
        high_risk_feat = (
            features_df[features_df["msno"].isin(high_risk_msno)]
            .set_index("msno")[available]
            .fillna(0)
        )

        uplift_scores = compute_uplift(high_risk_feat, available)
        uplift_df     = uplift_scores.reset_index()
        uplift_df.columns = ["msno", "uplift_score"]

        scored_df = scored_df.merge(uplift_df, on="msno", how="left")
        logger.info(
            f"Uplift scores added | "
            f"Mean: {uplift_df['uplift_score'].mean():.3f} | "
            f"Persuadable (>0.05): {(uplift_df['uplift_score'] > 0.05).sum():,}"
        )
    except FileNotFoundError:
        logger.warning("Uplift models not found — skipping.")
        scored_df["uplift_score"] = None

    return scored_df


def main() -> None:
    logger.info("=" * 55)
    logger.info("SCORING PIPELINE")
    logger.info("=" * 55)

    features_df = load_features()

    logger.info("[1/3] Scoring all users...")
    scored_df = score_all_users(features_df)

    logger.info("[2/3] Computing SHAP for high-risk users...")
    scored_df = add_shap(scored_df, features_df)

    logger.info("[3/3] Computing uplift scores for high-risk users...")
    scored_df = add_uplift(scored_df, features_df)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scored_path = EXPORTS_DIR / "scored_users.csv"
    scored_df.to_csv(scored_path, index=False)

    logger.success(
        f"Scoring complete.\n"
        f"  Total users scored  : {len(scored_df):,}\n"
        f"  High-risk flagged   : {int(scored_df['churn_flag'].sum()):,}\n"
        f"  Saved               : {scored_path}\n"
        f"  Next                : make api"
    )


if __name__ == "__main__":
    main()