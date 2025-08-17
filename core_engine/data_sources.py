
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_ticks(num_days=60, symbols=("SPX",)):
    """Week 0–2: create simple synthetic options-like rows for a few dates."""
    np.random.seed(42)
    data = []
    start_date = (datetime.utcnow() - timedelta(days=num_days)).date()
    for d in range(num_days):
        as_of = start_date + timedelta(days=d)
        # Only keep weekdays to feel like market days
        if as_of.weekday() >= 5:
            continue
        spot = 5000 + np.random.randn() * 25  # mock underlying level
        for strike in range(4800, 5201, 25):
            for opt_type in ("call", "put"):
                last_price = max(0.1, np.random.normal(6, 2))
                iv = float(np.clip(np.random.normal(0.25, 0.06), 0.10, 0.75))
                # rough delta directionality by moneyness + type
                m = (spot - strike) / spot
                base_delta = 0.5 if opt_type == "call" else -0.5
                delta = float(np.clip(base_delta + m, -1, 1))
                expiry = (datetime.combine(as_of, datetime.min.time()) + timedelta(days=7)).date()
                data.append({
                    "as_of_date": as_of.isoformat(),
                    "symbol": f"SPX_{strike}_{as_of.strftime('%Y%m%d')}",
                    "underlying": "SPX",
                    "option_type": opt_type,
                    "strike": float(strike),
                    "last_price": round(last_price, 2),
                    "iv": round(iv, 4),
                    "delta": round(delta, 4),
                    "expiry": expiry.isoformat()
                })
    return pd.DataFrame(data)

def save_mock_data(path="data/training_dataset.parquet"):
    df = generate_mock_ticks()
    # For training we also need a proxy for the best price before expiry (target construction downstream)
    # Create a noisy "max_price_until_expiry" that is between 0.8x and 1.6x the entry price
    rng = np.random.default_rng(123)
    df["max_price_until_expiry"] = df["last_price"] * rng.uniform(0.8, 1.6, size=len(df))
    df.to_parquet(path, index=False)
    print(f"✅ Saved mock data to {path} ({len(df):,} rows)")
