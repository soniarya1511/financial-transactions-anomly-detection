"""Single source of truth for paths and the model's input contract.

The contract is FROZEN: it mirrors exactly how `night_4.h5` was trained in 2025
(see docs/DECISIONS.md, ADR-001). Changing anything here without retraining breaks the model.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "docs" / "figures"

TRAIN_CSV = DATA_DIR / "non_fraud_night_1.csv.gz"   # 100k synthetic normal users (training population)
FRAUD_CSV = DATA_DIR / "fraud_night_1.csv"           # 2.5k synthetic fraud users (never seen in training)

MODEL_PATH = MODEL_DIR / "night_4.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODERS_PATH = MODEL_DIR / "label_encoders.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

ID_COL = "user_id"
DROP_COLS = ["user_id", "account_number", "transaction_id", "Unnamed: 0"]

# Columns that were label-encoded at training time (string columns of the training CSV).
# `phone_number_prefix` is NOT here: pandas parsed "+44" as the integer 44, so it was scaled as a number.
CATEGORICAL_COLS = [
    "gender", "country", "city", "account_creation_date", "account_type", "subscription_plan",
    "email_domain", "device_type", "browser_type", "OS", "payment_methods_used", "most_active_community",
]

# Reproduces `train_test_split(X, test_size=0.2, random_state=42)` from the training notebook,
# so the 20k validation rows here are the same rows Keras used as validation_data.
SPLIT_SEED = 42
VAL_FRACTION = 0.2

# Semantic groups (from the generator's own comments), used for EDA and attribution.
FEATURE_GROUPS = {
    "profile": ["age", "gender", "latitude", "longitude", "country", "city", "account_creation_date",
                "account_type", "subscription_plan", "email_domain", "phone_number_prefix"],
    "behaviour": ["session_duration_avg", "time_spent_per_action", "number_of_sessions_per_day",
                  "device_type", "browser_type", "OS", "most_active_hours",
                  "weekend_vs_weekday_activity_ratio", "login_frequency_per_week",
                  "failed_login_attempts", "password_reset_requests"],
    "transactions": ["total_transactions", "avg_transaction_value", "max_transaction_value",
                     "min_transaction_value", "number_of_refunds", "chargeback_ratio",
                     "number_of_failed_transactions", "payment_methods_used"],
    "social": ["number_of_friends", "number_of_messages_sent", "number_of_posts",
               "most_active_community", "engagement_score"],
    "network": ["IP_address_diversity", "VPN_or_Proxy_usage", "device_switching_frequency",
                "MAC_address_changes", "number_of_sessions_from_unusual_locations",
                "average_network_latency", "packet_loss_rate"],
    "security": ["number_of_failed_captcha_attempts", "multi_factor_authentication_enabled",
                 "account_suspension_count", "fraud_report_count", "password_change_frequency",
                 "device_trust_score"],
    # Pre-computed risk scores: label-like, see docs/REPORT.md "Leakage".
    "precomputed_scores": ["outlier_score", "transaction_pattern_anomaly_score",
                           "time_series_anomaly_score", "fraud_risk_score"],
}
