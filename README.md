# Churn Prevention System

An end-to-end ML pipeline for detecting, explaining, and acting on subscription churn risk — built on real KKBox behavioral data with synthetic intervention layers.

---

## What This Does

Most churn projects stop at prediction. This one closes the loop:

**Detect** → **Explain** → **Act** → **Measure** → **Retrain**

1. **SQL warehouse** — DuckDB with sessionization, rolling 30/60/90d metrics, behavioral cohorts
2. **Change point detection** — PELT algorithm finds *when* a user's behavior shifted, not just *that* it did
3. **Churn classifier** — XGBoost with 5-fold CV, optimized for AUC-PR (class imbalance ~94/6)
4. **Uplift model** — T-learner meta-learner separates persuadable users from non-persuadable ones
5. **LLM explanation layer** — SHAP values + change points → plain English insight via Groq API
6. **Intervention selection** — uplift-gated action plan per user (discount / re-onboarding / outreach)
7. **Drift monitoring** — Evidently-based feature drift detection with auto-retraining trigger
8. **FastAPI** — live endpoints for churn risk, intervention recommendation, and drift status

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/churn-prevention-system
cd churn-prevention-system
make setup

# 2. Add your LLM API key to .env (optional but recommended for explanations)
echo "GROQ_API_KEY=your_key_here" >> .env

# 3. Download KKBox data
kaggle competitions download -c kkbox-churn-prediction-challenge
unzip kkbox-churn-prediction-challenge.zip -d data/raw/kkbox

# 4. Generate synthetic extension
python generate_synthetic_data.py --sample_users 50000

# 5. Run full pipeline (ingest → features → models → action_plan.csv)
python -m pipelines.full_pipeline

# 6. (Optional) Run scoring-only pipeline to refresh scored_users.csv for the API
python -m pipelines.scoring_pipeline

# 7. Start API + dashboard
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/         (React dashboard)
# → http://localhost:8000/docs     (API docs)
```

---

## API & Dashboard

**Dashboard**

- `/` — React-based dashboard showing:
  - Top high-risk users ranked by churn probability.
  - Segment insights by risk tier and behavioral cohort.
  - Simple monitoring and model insight views.

**API Endpoints**

| Endpoint | Description |
|---|---|
| `GET /users/churn-risk?msno=...` | Churn probability + plain-English explanation and recommended action |
| `GET /monitoring/drift-report` | Feature drift status + retraining flag |

**Example response — `/users/{msno}/churn-risk`:**
```json
{
  "churn_probability": 0.83,
  "churn_flag": true,
  "explanation": "This user's engagement dropped 42% after March 3rd and they've submitted 3 support tickets in the last 30 days, 2 of which were cancellation inquiries. Their plan was downgraded last month.",
  "change_point_detected": true,
  "change_point_date": "2017-03-03"
}
```

---

## Project Structure (Final)

```
sql/              Feature engineering in SQL (sessionization, rolling metrics, marts)
src/
  ingestion/      CSV → DuckDB warehouse
  features/       SQL orchestration + (optional) PELT change point loading
  models/         XGBoost churn classifier + T-learner uplift model
  explainability/ SHAP values + LLM narration
  interventions/  Uplift-gated action selection
  monitoring/     Drift detection + retraining trigger
app/              FastAPI serving layer
pipelines/        End-to-end, scoring-only, and retraining pipelines
notebooks/        EDA, model development, example outputs (for recruiters)
```

---

## Key Design Choices

**Why uplift modeling?** Standard churn scores don't distinguish between users who would churn regardless and users who can be saved by outreach. Uplift modeling (T-learner) outputs *incremental* treatment effect — so we only contact users where intervention is actually likely to help, reducing wasted retention spend.

**Why PELT change points?** SHAP tells you which features matter. PELT tells you *when* the user's behavior regime shifted. Combining both gives the LLM a temporal anchor: "usage dropped 42% after March 3rd" is more actionable than "usage is low."

**Why DuckDB?** Zero infrastructure — runs locally on 3GB+ CSVs with full SQL window function support. Production swap to Snowflake/BigQuery is trivial.

---

## Data

Real: [KKBox Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) (Kaggle)

Synthetic extension: `generate_synthetic_data.py` — seeds distributions from KKBox and generates feature events, support tickets, A/B intervention assignments, and outcome labels for the uplift model.
