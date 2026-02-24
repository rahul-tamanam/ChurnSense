"""
SHAP Explainer
--------------
Computes per-user SHAP values using the trained churn classifier.
Returns a DataFrame with shap_<feature> columns aligned to the scored users.
"""
import shap
import pandas as pd
import numpy as np
from loguru import logger
from src.utils.config import CHURN_FEATURES
from src.models.model_registry import load_model


def explain(X: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Compute SHAP values for all rows in X.

    Args:
        X: feature DataFrame with msno as index (must contain CHURN_FEATURES columns)
        model: optional pre-loaded model (loads from registry if None)

    Returns:
        DataFrame with shap_<feature> columns, same index as X
    """
    if model is None:
        model = load_model("churn_classifier")

    available = [c for c in CHURN_FEATURES if c in X.columns]
    missing   = set(CHURN_FEATURES) - set(available)
    if missing:
        logger.warning(f"SHAP: missing features (filling with 0): {missing}")

    X_model = X[available].copy().fillna(0)

    logger.info(f"Computing SHAP values for {len(X_model):,} users...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_model)

    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{c}" for c in available],
        index=X.index,
    )

    logger.info("SHAP computation complete")
    return shap_df


def top_factors(shap_row: pd.Series, n: int = 5) -> list[dict]:
    """
    Given a single row of SHAP values (shap_<feature> columns),
    return the top N features by absolute impact as a list of dicts.

    Used by the API to return human-readable risk factors per user.
    """
    items = [
        (col.replace("shap_", ""), val)
        for col, val in shap_row.items()
        if col.startswith("shap_")
    ]
    items.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature":    feat,
            "shap_value": round(float(val), 4),
            "direction":  "risk_increase" if val > 0 else "risk_decrease",
        }
        for feat, val in items[:n]
    ]