"""
Synthetic Data Generator for SaaS Churn Prevention System
----------------------------------------------------------
Reads KKBox real data distributions and generates 4 synthetic tables:
  1. feature_events.csv       - SaaS-style feature usage events
  2. support_tickets.csv      - Support interactions (correlated with churn)
  3. interventions.csv        - A/B retention intervention assignments
  4. outcomes.csv             - Intervention outcomes (for uplift model)

Usage:
    python generate_synthetic_data.py \
        --kkbox_dir data/raw/kkbox \
        --output_dir data/raw/synthetic \
        --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kkbox_dir",   default="data/raw/kkbox",      help="Folder with KKBox CSVs")
    parser.add_argument("--output_dir",  default="data/synthetic",   help="Output folder for synthetic tables")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--sample_users", type=int, default=50_000,     help="Cap users for speed (set 0 for all)")
    return parser.parse_args()

# ── LOADERS ──────────────────────────────────────────────────────────────────

def load_kkbox(kkbox_dir: str, sample_users: int, rng: np.random.Generator):
    print("Loading KKBox members...")
    members = pd.read_csv(os.path.join(kkbox_dir, "members_v3.csv"))

    print("Loading churn labels...")
    labels_path = os.path.join(kkbox_dir, "train_v2.csv")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(kkbox_dir, "train.csv")
    labels = pd.read_csv(labels_path)

    # merge so we only work with labelled users
    users = members.merge(labels, on="msno", how="inner")

    if sample_users and sample_users < len(users):
        users = users.sample(n=sample_users, random_state=int(rng.integers(1e6)))
        print(f"  Sampled {sample_users:,} users from {len(members):,}")
    else:
        print(f"  Using all {len(users):,} labelled users")

    print("Loading user_logs (this may take a moment)...")
    logs_path = os.path.join(kkbox_dir, "user_logs_v2.csv")
    if not os.path.exists(logs_path):
        logs_path = os.path.join(kkbox_dir, "user_logs.csv")

    # Only load logs for our sampled users to keep memory sane
    user_set = set(users["msno"].tolist())
    chunks = []
    for chunk in pd.read_csv(logs_path, chunksize=500_000):
        chunk = chunk[chunk["msno"].isin(user_set)]
        if len(chunk):
            chunks.append(chunk)
    logs = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(logs):,} log rows for sampled users")

    print("Loading transactions...")
    txn_path = os.path.join(kkbox_dir, "transactions_v2.csv")
    if not os.path.exists(txn_path):
        txn_path = os.path.join(kkbox_dir, "transactions.csv")
    txn = pd.read_csv(txn_path)
    txn = txn[txn["msno"].isin(user_set)]
    print(f"  Loaded {len(txn):,} transaction rows")

    return users, logs, txn

# ── HELPERS ──────────────────────────────────────────────────────────────────

def user_activity_stats(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user engagement stats from real logs to seed synthetic generation."""
    logs["date"] = pd.to_datetime(logs["date"].astype(str), format="%Y%m%d", errors="coerce")
    stats = logs.groupby("msno").agg(
        total_seconds_mean  = ("total_secs", "mean"),
        total_seconds_std   = ("total_secs", "std"),
        num_25_mean         = ("num_25", "mean"),
        active_days         = ("date", "count"),
        first_log           = ("date", "min"),
        last_log            = ("date", "max"),
    ).reset_index()
    stats["total_seconds_std"] = stats["total_seconds_std"].fillna(50)
    stats["tenure_days"] = (stats["last_log"] - stats["first_log"]).dt.days.clip(lower=1)
    return stats

def last_transaction_date(txn: pd.DataFrame) -> pd.Series:
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"].astype(str), format="%Y%m%d", errors="coerce")
    return txn.groupby("msno")["transaction_date"].max()

# ── TABLE 1: FEATURE EVENTS ───────────────────────────────────────────────────

FEATURES = [
    "playlist_create", "song_download", "social_share", "discovery_mode",
    "radio_mode", "lyrics_view", "artist_follow", "album_view",
    "search_use",     "settings_change",
]

def generate_feature_events(users: pd.DataFrame, logs_stats: pd.DataFrame,
                             rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate SaaS-style feature usage events.
    Churners get declining usage in final 30 days; non-churners stay stable.
    """
    print("Generating feature_events...")
    merged = users.merge(logs_stats, on="msno", how="left")
    merged["last_log"] = merged["last_log"].fillna(pd.Timestamp("2017-03-01"))
    merged["tenure_days"] = merged["tenure_days"].fillna(30)

    records = []
    for row in merged.itertuples(index=False):
        msno      = row.msno
        is_churn  = row.is_churn
        end_date  = row.last_log
        tenure    = int(min(row.tenure_days, 180))
        start_date = end_date - timedelta(days=tenure)

        # Base event rate: 0-5 feature events per day
        base_rate = rng.integers(1, 6)

        dates = pd.date_range(start_date, end_date, freq="D")
        for i, d in enumerate(dates):
            # Churners: usage decays in last 30 days
            day_from_end = (end_date - d).days
            if is_churn and day_from_end <= 30:
                decay = max(0.1, 1 - (30 - day_from_end) / 30 * 0.85)
            else:
                decay = 1.0

            n_events = rng.poisson(base_rate * decay)
            features_today = rng.choice(FEATURES, size=n_events, replace=True)
            for feat in features_today:
                records.append({
                    "msno":         msno,
                    "event_date":   d.date(),
                    "feature_name": feat,
                    "session_duration_sec": int(rng.exponential(180)),
                })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} feature events")
    return df

# ── TABLE 2: SUPPORT TICKETS ──────────────────────────────────────────────────

TICKET_CATEGORIES = ["billing", "playback_error", "account_access",
                     "cancellation_inquiry", "feature_request", "other"]
TICKET_SENTIMENTS = ["negative", "neutral", "positive"]
RESOLUTIONS       = ["resolved", "unresolved", "escalated"]

def generate_support_tickets(users: pd.DataFrame, logs_stats: pd.DataFrame,
                              rng: np.random.Generator) -> pd.DataFrame:
    """
    Churners have 2-4x higher ticket rate and more negative sentiment.
    Billing / cancellation tickets spike in last 14 days before churn.
    """
    print("Generating support_tickets...")
    merged = users.merge(logs_stats, on="msno", how="left")
    merged["last_log"] = merged["last_log"].fillna(pd.Timestamp("2017-03-01"))
    merged["tenure_days"] = merged["tenure_days"].fillna(30)

    records = []
    ticket_id = 1
    for row in merged.itertuples(index=False):
        msno     = row.msno
        is_churn = row.is_churn
        end_date = row.last_log
        tenure   = int(min(row.tenure_days, 180))
        start    = end_date - timedelta(days=tenure)

        # Overall ticket count
        base_tickets = rng.integers(0, 3) if not is_churn else rng.integers(1, 7)

        for _ in range(base_tickets):
            # Churners: cluster tickets near end
            if is_churn:
                days_ago = int(rng.exponential(20))  # skewed recent
            else:
                days_ago = int(rng.uniform(0, tenure))
            ticket_date = end_date - timedelta(days=days_ago)
            if ticket_date < start:
                ticket_date = start

            # Category: churners get more billing/cancellation
            if is_churn and days_ago <= 14:
                cat_weights = [0.35, 0.15, 0.15, 0.25, 0.05, 0.05]
            else:
                cat_weights = [0.20, 0.25, 0.20, 0.10, 0.15, 0.10]

            cat      = rng.choice(TICKET_CATEGORIES, p=cat_weights)
            sent_w   = [0.60, 0.30, 0.10] if is_churn else [0.15, 0.50, 0.35]
            sent     = rng.choice(TICKET_SENTIMENTS, p=sent_w)
            resolved = rng.choice(RESOLUTIONS, p=[0.50, 0.30, 0.20] if is_churn else [0.75, 0.15, 0.10])
            ttf      = int(rng.exponential(24)) if resolved == "resolved" else None  # time-to-fix hours

            records.append({
                "ticket_id":         ticket_id,
                "msno":              msno,
                "ticket_date":       ticket_date.date(),
                "category":          cat,
                "sentiment":         sent,
                "resolution_status": resolved,
                "time_to_fix_hours": ttf,
            })
            ticket_id += 1

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} support tickets")
    return df

# ── TABLE 3: INTERVENTIONS ────────────────────────────────────────────────────

INTERVENTION_TYPES = ["discount_offer", "feature_highlight_email",
                      "personal_outreach", "re_onboarding_flow", "no_treatment"]

def generate_interventions(users: pd.DataFrame, logs_stats: pd.DataFrame,
                            rng: np.random.Generator) -> pd.DataFrame:
    """
    A/B assignment. ~70% of users get some intervention, 30% control.
    Intervention timing: 7-21 days before last log (realistic retention window).
    """
    print("Generating interventions...")
    merged = users.merge(logs_stats, on="msno", how="left")
    merged["last_log"] = merged["last_log"].fillna(pd.Timestamp("2017-03-01"))

    records = []
    for row in merged.itertuples(index=False):
        # Assign treatment
        if rng.random() < 0.30:
            treatment = "no_treatment"
        else:
            treatment = rng.choice(INTERVENTION_TYPES[:-1])  # exclude no_treatment

        days_before = int(rng.uniform(7, 22))
        intervention_date = row.last_log - timedelta(days=days_before)

        opened = None
        clicked = None
        if treatment in ("discount_offer", "feature_highlight_email", "re_onboarding_flow"):
            opened  = bool(rng.random() < (0.25 if row.is_churn else 0.45))
            clicked = bool(rng.random() < (0.10 if row.is_churn else 0.30)) if opened else False

        records.append({
            "msno":              row.msno,
            "intervention_type": treatment,
            "intervention_date": intervention_date.date(),
            "email_opened":      opened,
            "email_clicked":     clicked,
            "is_churn":          row.is_churn,  # keep for outcome generation join
        })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} intervention assignments")
    return df

# ── TABLE 4: OUTCOMES ─────────────────────────────────────────────────────────

# Uplift by intervention type (incremental churn reduction probability)
UPLIFT_PARAMS = {
    "discount_offer":          0.18,
    "feature_highlight_email": 0.09,
    "personal_outreach":       0.22,
    "re_onboarding_flow":      0.14,
    "no_treatment":            0.00,
}

def generate_outcomes(interventions: pd.DataFrame,
                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Outcome = did user renew after intervention?
    Churners who received high-uplift treatment have a chance of being saved.
    Non-churners renew regardless.
    """
    print("Generating outcomes...")
    records = []
    for row in interventions.itertuples(index=False):
        uplift   = UPLIFT_PARAMS.get(row.intervention_type, 0)
        saved    = False

        if row.is_churn:
            # Was the intervention effective?
            saved = bool(rng.random() < uplift)
            renewed = saved
        else:
            renewed = bool(rng.random() < 0.92)  # non-churners almost always renew

        records.append({
            "msno":              row.msno,
            "intervention_type": row.intervention_type,
            "renewed":           renewed,
            "was_saved":         saved,      # churner who converted
            "observed_churn":    not renewed,
        })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df):,} outcome records")
    # Save rate sanity check
    treated = df[df["intervention_type"] != "no_treatment"]
    print(f"  Treated save rate: {treated['was_saved'].mean():.1%}")
    print(f"  Control save rate: {df[df['intervention_type']=='no_treatment']['was_saved'].mean():.1%}")
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load real KKBox data
    users, logs, txn = load_kkbox(args.kkbox_dir, args.sample_users, rng)

    # Derive activity stats from real logs (seeds synthetic distributions)
    logs_stats = user_activity_stats(logs)

    # Generate synthetic tables
    feature_events  = generate_feature_events(users, logs_stats, rng)
    support_tickets = generate_support_tickets(users, logs_stats, rng)
    interventions   = generate_interventions(users, logs_stats, rng)
    outcomes        = generate_outcomes(interventions, rng)

    # Drop the is_churn helper column from interventions before saving
    interventions = interventions.drop(columns=["is_churn"])

    # Save
    paths = {
        "feature_events.csv":  feature_events,
        "support_tickets.csv": support_tickets,
        "interventions.csv":   interventions,
        "outcomes.csv":        outcomes,
    }
    for fname, df in paths.items():
        path = os.path.join(args.output_dir, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {fname}: {len(df):,} rows → {path}")

    print("\nDone. Synthetic data ready.")
    print(f"Real KKBox tables:     {args.kkbox_dir}/")
    print(f"Synthetic extensions:  {args.output_dir}/")

if __name__ == "__main__":
    main()