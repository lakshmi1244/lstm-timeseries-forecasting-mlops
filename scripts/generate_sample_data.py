"""Generate a synthetic daily spending time series with injected anomalies.

This lets you try out the full train/evaluate pipeline without needing
real financial data. It is for demonstration purposes only.

Usage:
    python scripts/generate_sample_data.py --output data/transactions.csv --days 1000
"""
import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic financial time series")
    parser.add_argument("--output", type=str, default="data/transactions.csv")
    parser.add_argument("--days", type=int, default=1000)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    t = np.arange(args.days)
    trend = 0.02 * t
    weekly_seasonality = 15 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 5, size=args.days)
    spending = 200 + trend + weekly_seasonality + noise

    is_anomaly = rng.random(args.days) < args.anomaly_rate
    spending[is_anomaly] += rng.normal(150, 40, size=is_anomaly.sum()) * rng.choice([-1, 1], size=is_anomaly.sum())

    dates = pd.date_range("2023-01-01", periods=args.days, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "daily_spending": spending,
        "is_anomaly": is_anomaly.astype(int),
    })

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} ({is_anomaly.sum()} injected anomalies).")


if __name__ == "__main__":
    main()
