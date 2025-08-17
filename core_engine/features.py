
import numpy as np
import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Week 2–4: feature engineering (days to expiry, moneyness, scaled delta)."""
    df = df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry"]).dt.tz_localize(None)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.tz_localize(None)
    df["days_to_expiry"] = (df["expiry_date"] - df["as_of_date"]).dt.days.clip(lower=0)

    # Moneyness proxy using a synthetic underlying of 5000; in real pipeline replace with true spot
    # We can try to infer spot from strike and delta sign, but keep it simple here.
    assumed_spot = 5000.0
    df["moneyness"] = df["strike"] / assumed_spot

    # Normalize delta to [0,1]
    df["delta_scaled"] = (df["delta"] + 1.0) / 2.0

    # Keep a compact feature set
    keep = [
        "symbol","underlying","option_type","strike","last_price","iv","delta","delta_scaled",
        "days_to_expiry","moneyness","as_of_date","expiry","max_price_until_expiry"
    ]
    return df[keep]
