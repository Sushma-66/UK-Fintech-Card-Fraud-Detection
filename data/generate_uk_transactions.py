"""
Generate a synthetic dataset of UK debit/credit card transactions with fraud labels.

Why synthetic?
    Real bank transaction data is confidential. The well-known Kaggle credit card
    dataset is European and its features are anonymised (V1..V28), so you cannot
    explain *why* a transaction looks fraudulent. Here every column has a real
    meaning, and the fraud patterns are based on how card fraud works in the UK
    (e.g. most losses come from "card not present" / remote purchase fraud,
    contactless payments are capped at GBP 100, fraudsters move fast).

The numbers are made up. Do not present them as real UK bank statistics.

Run:
    python data/generate_uk_transactions.py
Output:
    data/uk_card_transactions.csv.gz
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 8_000
N_TRANSACTIONS = 250_000
FRAUD_RATE = 0.0025          # 0.25% of transactions are fraud (very imbalanced, like real life)
START = pd.Timestamp("2026-09-01")
DAYS = 30

rng = np.random.default_rng(SEED)

UK_REGIONS = [
    "London", "South East", "South West", "East of England", "West Midlands",
    "East Midlands", "Yorkshire and the Humber", "North West", "North East",
    "Wales", "Scotland", "Northern Ireland",
]
REGION_WEIGHTS = np.array([16, 14, 8, 9, 9, 7, 8, 11, 4, 5, 8, 3], dtype=float)
REGION_WEIGHTS /= REGION_WEIGHTS.sum()

# merchant category -> (typical amount in GBP (lognormal median), share of legit spend, share of fraud)
MERCHANTS = {
    "grocery":         (25,  0.26, 0.04),
    "fuel":            (45,  0.08, 0.03),
    "restaurants":     (30,  0.12, 0.04),
    "online_retail":   (40,  0.15, 0.24),
    "electronics":     (150, 0.03, 0.17),
    "travel":          (180, 0.03, 0.10),
    "entertainment":   (20,  0.07, 0.04),
    "utilities":       (70,  0.06, 0.01),
    "gambling":        (40,  0.02, 0.10),
    "cash_withdrawal": (60,  0.06, 0.08),
    "fashion":         (55,  0.08, 0.10),
    "digital_goods":   (12,  0.04, 0.05),
}
MERCHANT_NAMES = list(MERCHANTS)
LEGIT_M_P = np.array([MERCHANTS[m][1] for m in MERCHANT_NAMES]); LEGIT_M_P /= LEGIT_M_P.sum()
FRAUD_M_P = np.array([MERCHANTS[m][2] for m in MERCHANT_NAMES]); FRAUD_M_P /= FRAUD_M_P.sum()

CARD_TYPES = ["debit", "credit"]


def make_customers(n):
    """One row per customer: their home region, how long they've banked with us, typical spend."""
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "home_region": rng.choice(UK_REGIONS, size=n, p=REGION_WEIGHTS),
        "card_type": rng.choice(CARD_TYPES, size=n, p=[0.7, 0.3]),
        "account_age_months": np.clip(rng.gamma(2.5, 30, size=n).round(), 1, 480).astype(int),
        "spend_multiplier": rng.lognormal(0, 0.35, size=n),  # some people spend more than others
    })


def pick_channel(merchants, is_fraud):
    """Choose how the card was used. Fraud is mostly 'card not present' (online)."""
    n = len(merchants)
    channel = np.empty(n, dtype=object)
    for i, m in enumerate(merchants):
        if m == "cash_withdrawal":
            channel[i] = "atm"
        elif m in ("online_retail", "digital_goods", "gambling"):
            channel[i] = rng.choice(["online", "mobile_wallet"], p=[0.8, 0.2])
        elif is_fraud:
            channel[i] = rng.choice(["online", "contactless", "chip_pin", "mobile_wallet"],
                                    p=[0.62, 0.18, 0.12, 0.08])
        else:
            channel[i] = rng.choice(["online", "contactless", "chip_pin", "mobile_wallet"],
                                    p=[0.22, 0.43, 0.20, 0.15])
    return channel


def make_transactions(customers, n, is_fraud):
    cust = customers.sample(n=n, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    merchants = rng.choice(MERCHANT_NAMES, size=n, p=FRAUD_M_P if is_fraud else LEGIT_M_P)

    # ---- amount (GBP) ----
    base = np.array([MERCHANTS[m][0] for m in merchants])
    amount = base * cust["spend_multiplier"].values * rng.lognormal(0, 0.6, size=n)
    if is_fraud:
        # fraudsters either "test" a card with tiny amounts, or go big
        style = rng.choice(["test", "normal", "big"], size=n, p=[0.18, 0.47, 0.35])
        amount = np.where(style == "test", rng.uniform(0.5, 3.0, size=n), amount)
        amount = np.where(style == "big", amount * rng.uniform(2.0, 6.0, size=n), amount)
    channel = pick_channel(merchants, is_fraud)
    # UK contactless limit is GBP 100 per transaction
    amount = np.where(channel == "contactless", np.minimum(amount, 100), amount)
    amount = np.round(np.clip(amount, 0.5, 9_500), 2)

    # ---- time of day ----
    if is_fraud:
        hour = rng.choice(24, size=n, p=_hour_profile(fraud=True))
    else:
        hour = rng.choice(24, size=n, p=_hour_profile(fraud=False))
    day = rng.integers(0, DAYS, size=n)
    ts = START + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") \
        + pd.to_timedelta(rng.integers(0, 3600, size=n), "s")

    # ---- where it happened ----
    international = rng.random(n) < (0.22 if is_fraud else 0.04)
    same_region = rng.random(n) < (0.45 if is_fraud else 0.82)
    other_regions = rng.choice(UK_REGIONS, size=n, p=REGION_WEIGHTS)
    txn_region = np.where(same_region, cust["home_region"].values, other_regions)
    txn_region = np.where(international, "Overseas", txn_region)
    distance = np.where(
        international, rng.uniform(300, 9_000, size=n),
        np.where(txn_region == cust["home_region"].values,
                 rng.exponential(6, size=n), rng.uniform(30, 600, size=n)))
    # online purchases: distance is the billing-to-IP distance, often small for genuine customers
    distance = np.round(distance, 1)

    # ---- behaviour signals the bank would compute ----
    txn_count_1h = rng.poisson(2.2 if is_fraud else 0.4, size=n)
    txn_count_24h = txn_count_1h + rng.poisson(5.5 if is_fraud else 3.0, size=n)
    new_merchant = rng.random(n) < (0.78 if is_fraud else 0.22)
    avg_spend = (35 * cust["spend_multiplier"].values * rng.lognormal(0, 0.2, size=n)).round(2)
    failed_pin = np.where(np.isin(channel, ["chip_pin", "atm"]),
                          rng.poisson(0.8 if is_fraud else 0.03, size=n), 0)
    # 3-D Secure (Strong Customer Authentication) - fraud tends to get through when it is NOT used
    sca_used = np.where(channel == "online", rng.random(n) < (0.30 if is_fraud else 0.70), False)

    return pd.DataFrame({
        "customer_id": cust["customer_id"].values,
        "timestamp": ts,
        "amount_gbp": amount,
        "merchant_category": merchants,
        "channel": channel,
        "card_type": cust["card_type"].values,
        "home_region": cust["home_region"].values,
        "transaction_region": txn_region,
        "is_international": international.astype(int),
        "distance_from_home_km": distance,
        # some fraud comes from newly opened accounts (e.g. "first-party"/application fraud)
        "account_age_months": np.where(rng.random(n) < (0.30 if is_fraud else 0.0),
                                       rng.integers(1, 6, size=n),
                                       cust["account_age_months"].values).astype(int),
        "customer_avg_spend_gbp": avg_spend,
        "txn_count_last_1h": txn_count_1h,
        "txn_count_last_24h": txn_count_24h,
        "is_new_merchant": new_merchant.astype(int),
        "failed_pin_attempts": failed_pin,
        "sca_authenticated": sca_used.astype(int),
        "is_fraud": int(is_fraud),
    })


def _hour_profile(fraud):
    hours = np.arange(24)
    if fraud:
        # spread across the day, with a bump overnight (00:00-05:00)
        w = 1 + 1.8 * ((hours <= 5) | (hours >= 23))
    else:
        # normal people shop in the day and evening, very little at night
        w = np.exp(-((hours - 13) ** 2) / 30) + 0.6 * np.exp(-((hours - 19) ** 2) / 8) + 0.03
    return w / w.sum()


def main():
    customers = make_customers(N_CUSTOMERS)
    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    legit = make_transactions(customers, N_TRANSACTIONS - n_fraud, is_fraud=False)
    fraud = make_transactions(customers, n_fraud, is_fraud=True)

    # Make it harder (and more realistic): 20% of fraud rows "look normal" -
    # we copy the behaviour columns from random genuine transactions.
    disguise = rng.random(len(fraud)) < 0.20
    donor = legit.sample(n=int(disguise.sum()), random_state=SEED).reset_index(drop=True)
    cols = ["txn_count_last_1h", "txn_count_last_24h", "is_new_merchant", "is_international",
            "distance_from_home_km", "transaction_region", "home_region", "sca_authenticated"]
    fraud.loc[disguise, cols] = donor[cols].values

    df = pd.concat([legit, fraud], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])

    out = Path(__file__).resolve().parent / "uk_card_transactions.csv.gz"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} transactions ({df['is_fraud'].sum():,} fraud, "
          f"{df['is_fraud'].mean():.3%}) to {out}")


if __name__ == "__main__":
    main()
