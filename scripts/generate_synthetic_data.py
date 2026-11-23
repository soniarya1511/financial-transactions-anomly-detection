"""Regenerate synthetic user populations with the same distributions as the hackathon data.

This is a faithful, seeded port of cells 1-2 of `legacy/train_autoencoder_original.ipynb`.
The committed CSVs in `data/` are the ORIGINAL files the shipped model was trained and
evaluated on; this script exists to document the data-generating process and to create
fresh samples (e.g. for drift tests). Its output will NOT be byte-identical to them
because the original run was unseeded.

    python scripts/generate_synthetic_data.py --normal 100000 --fraud 2500 --seed 7 --out data/generated
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


def normal_users(n, rng, fake):
    u, ri, ch = rng.uniform, rng.integers, rng.choice
    return pd.DataFrame({
        "user_id": [fake.uuid4() for _ in range(n)],
        "age": ri(18, 70, n),
        "gender": ch(["Male", "Female", "Other"], n),
        "latitude": u(-90, 90, n),
        "longitude": u(-180, 180, n),
        "country": [fake.country() for _ in range(n)],
        "city": [fake.city() for _ in range(n)],
        "account_creation_date": [fake.date_this_decade() for _ in range(n)],
        "account_type": ch(["Personal", "Business", "Admin"], n, p=[0.8, 0.15, 0.05]),
        "subscription_plan": ch(["Free", "Basic", "Premium"], n, p=[0.6, 0.3, 0.1]),
        "email_domain": ch(["gmail.com", "yahoo.com", "outlook.com", "protonmail.com"], n),
        "phone_number_prefix": ch(["+1", "+44", "+91", "+81", "+33"], n),
        "session_duration_avg": u(5, 120, n),
        "time_spent_per_action": u(0.5, 15, n),
        "number_of_sessions_per_day": ri(1, 10, n),
        "device_type": ch(["PC", "Mobile", "Tablet"], n, p=[0.5, 0.4, 0.1]),
        "browser_type": ch(["Chrome", "Firefox", "Safari", "Edge", "Opera"], n),
        "OS": ch(["Windows", "Linux", "macOS", "Android", "iOS"], n),
        "most_active_hours": ri(0, 24, n),
        "weekend_vs_weekday_activity_ratio": u(0.5, 1.5, n),
        "login_frequency_per_week": ri(1, 20, n),
        "failed_login_attempts": ri(0, 5, n),
        "password_reset_requests": ri(0, 3, n),
        "total_transactions": ri(5, 500, n),
        "avg_transaction_value": u(10, 1000, n),
        "max_transaction_value": u(100, 10000, n),
        "min_transaction_value": u(1, 50, n),
        "number_of_refunds": ri(0, 5, n),
        "chargeback_ratio": u(0, 0.05, n),
        "number_of_failed_transactions": ri(0, 3, n),
        "payment_methods_used": ch(["Credit Card", "PayPal", "Crypto", "Bank Transfer"], n),
        "number_of_friends": ri(0, 500, n),
        "number_of_messages_sent": ri(0, 1000, n),
        "number_of_posts": ri(0, 300, n),
        "most_active_community": ch(["Tech", "Finance", "Gaming", "Health", "Education"], n),
        "engagement_score": u(0, 1, n),
        "IP_address_diversity": ri(1, 10, n),
        "VPN_or_Proxy_usage": ch([0, 1], n, p=[0.95, 0.05]),
        "device_switching_frequency": ri(0, 5, n),
        "MAC_address_changes": ri(0, 3, n),
        "number_of_sessions_from_unusual_locations": ri(0, 2, n),
        "average_network_latency": u(20, 300, n),
        "packet_loss_rate": u(0, 5, n),
        "number_of_failed_captcha_attempts": ri(0, 2, n),
        "multi_factor_authentication_enabled": ch([0, 1], n, p=[0.7, 0.3]),
        "account_suspension_count": np.zeros(n),
        "fraud_report_count": np.zeros(n),
        "password_change_frequency": ri(0, 5, n),
        "device_trust_score": u(0.7, 1.0, n),
        "outlier_score": u(0, 0.2, n),
        "transaction_pattern_anomaly_score": u(0, 0.2, n),
        "time_series_anomaly_score": u(0, 0.2, n),
        "fraud_risk_score": u(0, 0.1, n),
    })


def fraud_users(n, rng, fake):
    u, ri, ch = rng.uniform, rng.integers, rng.choice
    return pd.DataFrame({
        "user_id": [fake.uuid4() for _ in range(n)],
        "age": ri(18, 70, n),
        "gender": ch(["Male", "Female", "Other"], n),
        "latitude": u(-90, 90, n),
        "longitude": u(-180, 180, n),
        "country": [fake.country() for _ in range(n)],
        "city": [fake.city() for _ in range(n)],
        "account_creation_date": [fake.date_this_year() for _ in range(n)],
        "account_type": ch(["Personal", "Business", "Admin"], n, p=[0.6, 0.3, 0.1]),
        "subscription_plan": ch(["Free", "Basic", "Premium"], n, p=[0.7, 0.2, 0.1]),
        "email_domain": ch(["tempmail.com", "disposable.com", "randommail.net", "protonmail.com"], n),
        "phone_number_prefix": ch(["+999", "+888", "+777", "+666", "+555"], n),
        "session_duration_avg": u(1, 10, n),
        "time_spent_per_action": u(0.1, 2, n),
        "number_of_sessions_per_day": ri(10, 50, n),
        "device_type": ch(["PC", "Mobile", "Tablet"], n, p=[0.3, 0.6, 0.1]),
        "browser_type": ch(["Chrome", "Firefox", "Unknown", "BotBrowser"], n),
        "OS": ch(["Windows", "Linux", "macOS", "Android", "iOS"], n),
        "most_active_hours": ri(0, 24, n),
        "weekend_vs_weekday_activity_ratio": u(2, 5, n),
        "login_frequency_per_week": ri(20, 100, n),
        "failed_login_attempts": ri(5, 50, n),
        "password_reset_requests": ri(3, 10, n),
        "total_transactions": ri(50, 2000, n),
        "avg_transaction_value": u(500, 5000, n),
        "max_transaction_value": u(1000, 50000, n),
        "min_transaction_value": u(0.1, 100, n),
        "number_of_refunds": ri(5, 50, n),
        "chargeback_ratio": u(0.1, 0.9, n),
        "number_of_failed_transactions": ri(5, 30, n),
        "payment_methods_used": ch(["Stolen Card", "Crypto", "Anonymous Payment", "Gift Cards"], n),
        "number_of_friends": ri(0, 10, n),
        "number_of_messages_sent": ri(0, 5000, n),
        "number_of_posts": ri(0, 5, n),
        "most_active_community": ch(["Scam", "Dark Web", "Gambling", "Fake News"], n),
        "engagement_score": u(0, 0.2, n),
        "IP_address_diversity": ri(5, 50, n),
        "VPN_or_Proxy_usage": ch([0, 1], n, p=[0.2, 0.8]),
        "device_switching_frequency": ri(5, 20, n),
        "MAC_address_changes": ri(3, 10, n),
        "number_of_sessions_from_unusual_locations": ri(2, 10, n),
        "average_network_latency": u(300, 1000, n),
        "packet_loss_rate": u(5, 20, n),
        "number_of_failed_captcha_attempts": ri(2, 10, n),
        "multi_factor_authentication_enabled": ch([0, 1], n, p=[0.9, 0.1]),
        "account_suspension_count": ri(1, 5, n),
        "fraud_report_count": ri(1, 10, n),
        "password_change_frequency": ri(5, 20, n),
        "device_trust_score": u(0, 0.5, n),
        "outlier_score": u(0.5, 1.5, n),
        "transaction_pattern_anomaly_score": u(0.5, 1.5, n),
        "time_series_anomaly_score": u(0.5, 1.5, n),
        "fraud_risk_score": u(0.7, 1.0, n),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", type=int, default=100_000)
    ap.add_argument("--fraud", type=int, default=2_500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/generated"))
    a = ap.parse_args()
    rng, fake = np.random.default_rng(a.seed), Faker()
    Faker.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    normal_users(a.normal, rng, fake).to_csv(a.out / "normal_users.csv", index=False)
    fraud_users(a.fraud, rng, fake).to_csv(a.out / "fraud_users.csv", index=False)
    print(f"wrote {a.normal} normal + {a.fraud} fraud users to {a.out}/")
