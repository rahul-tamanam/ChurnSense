"""
LLM Narrator
-------------
Takes SHAP values + change point data for a user and generates:
  1. A plain-English explanation of WHY they're predicted to churn
  2. A recommended intervention action

Uses Groq's API (llama-3.3-70b-versatile) — fast, free-tier friendly.
"""
import json
import pandas as pd
from groq import Groq
from loguru import logger
from src.utils.config import GROQ_API_KEY


client = Groq(api_key=GROQ_API_KEY)

MODEL                 = "llama-3.3-70b-versatile"
TOP_N_FEATURES        = 5
MAX_USERS_FOR_NARRATION = 200  # limit batch size to avoid LLM rate limits

FEATURE_LABELS = {
    "usage_decay_30d":             "30-day usage decay",
    "usage_decay_60d":             "60-day usage decay",
    "ticket_count_30d":            "support tickets (last 30 days)",
    "pct_negative_sentiment":      "% negative support sentiment",
    "has_cancellation_inquiry":    "cancellation inquiry ticket",
    "plan_downgrade_flag":         "plan downgrade",
    "login_days_last_30d":         "login days (last 30 days)",
    "days_since_last_feature_use": "days since last feature use",
    "unique_features_used_30d":    "unique features used (30 days)",
    "payment_failure_count_90d":   "payment failures (90 days)",
    "change_point_detected":       "behavioral change point detected",
    "days_since_change_point":     "days since behavior changed",
    "pct_usage_change":            "% change in usage after shift",
}


def get_top_shap_factors(shap_row: dict, n: int = TOP_N_FEATURES) -> list[dict]:
    """Extract top N features by absolute SHAP value."""
    shap_items = [
        (k.replace("shap_", ""), v)
        for k, v in shap_row.items()
        if k.startswith("shap_") and v is not None
    ]
    shap_items.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature":   feat,
            "label":     FEATURE_LABELS.get(feat, feat.replace("_", " ")),
            "impact":    "increases churn risk" if val > 0 else "decreases churn risk",
            "magnitude": abs(val),
        }
        for feat, val in shap_items[:n]
    ]


def build_prompt(user_data: dict, shap_factors: list[dict], change_point: dict) -> str:
    cp_text = ""
    if change_point.get("change_point_detected"):
        pct  = change_point.get("pct_usage_change", 0)
        days = change_point.get("days_since_change_point", "unknown")
        date = change_point.get("change_point_date", "unknown date")
        cp_text = (
            f"\n- A significant behavioral shift was detected {days} days ago "
            f"(around {date}), with usage changing by {pct:.0%} after that point."
        )

    factors_text = "\n".join(
        f"- {f['label']}: {f['impact']} (magnitude: {f['magnitude']:.3f})"
        for f in shap_factors
    )

    return f"""You are a customer success analyst at a SaaS/music streaming company.

A machine learning model has flagged the following user as high churn risk.
Churn probability: {user_data.get('churn_prob', 0):.0%}

Top risk factors driving this prediction:
{factors_text}
{cp_text}

User context:
- Tenure: {user_data.get('tenure_days', 'unknown')} days
- Plan downgrade in last 90 days: {user_data.get('plan_downgrade_flag', False)}
- Open unresolved support tickets: {user_data.get('unresolved_ticket_count', 0)}
- Email engagement (last intervention): opened={user_data.get('email_opened', 'N/A')}, clicked={user_data.get('email_clicked', 'N/A')}

Your task:
1. Write a 2-3 sentence plain-English explanation of WHY this user is likely to churn. Be specific, not generic.
2. Recommend ONE concrete retention action from this list: [discount_offer, feature_highlight_email, personal_outreach, re_onboarding_flow, no_action].
3. Give a one-sentence justification for your recommended action.

Respond in this exact JSON format (no markdown, no code fences):
{{
  "explanation": "...",
  "recommended_action": "...",
  "action_justification": "..."
}}"""


def narrate(user_data: dict, shap_row: dict, change_point: dict) -> dict:
    """
    Generate LLM explanation and recommendation for a single user.

    Args:
        user_data:    dict with churn_prob, feature values, email engagement
        shap_row:     dict with shap_<feature> keys
        change_point: dict with change_point_detected, pct_usage_change, etc.

    Returns:
        dict with explanation, recommended_action, action_justification
    """
    factors = get_top_shap_factors(shap_row)
    prompt  = build_prompt(user_data, factors, change_point)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model adds them anyway
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)

    except Exception as e:
        logger.warning(f"LLM narration failed: {e}")
        return {
            "explanation":         "Unable to generate explanation.",
            "recommended_action":  "no_action",
            "action_justification": "LLM unavailable.",
        }


def narrate_batch(
    scored_df: pd.DataFrame,
    change_points_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run narration for a subset of high-risk users (top-N by churn_prob).
    Returns DataFrame with msno + explanation columns.
    """
    high_risk = scored_df[scored_df["churn_flag"] == 1].copy()
    if MAX_USERS_FOR_NARRATION:
        high_risk = (
            high_risk.sort_values("churn_prob", ascending=False)
            .head(MAX_USERS_FOR_NARRATION)
        )

    cp_lookup   = change_points_df.set_index("msno").to_dict("index") if change_points_df is not None else {}
    feat_lookup = features_df.set_index("msno").to_dict("index") if features_df is not None else {}

    results = []
    logger.info(f"Generating LLM narratives for {len(high_risk):,} high-risk users...")

    for row in high_risk.itertuples(index=False):
        msno         = row.msno
        user_data    = {"msno": msno, "churn_prob": row.churn_prob}
        user_data.update(feat_lookup.get(msno, {}))
        shap_row     = {k: getattr(row, k) for k in high_risk.columns if k.startswith("shap_")}
        change_point = cp_lookup.get(msno, {"change_point_detected": False})

        narrative = narrate(user_data, shap_row, change_point)
        results.append({"msno": msno, **narrative})

    return pd.DataFrame(results)
