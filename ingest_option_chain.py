
import pandas as pd
from core_engine.features import compute_features
from core_engine.utils import ensure_dir

if __name__ == "__main__":
    ensure_dir("data")
    df = pd.read_parquet("data/training_dataset.parquet")
    feats = compute_features(df)
    feats.to_parquet("data/features_dataset.parquet", index=False)
    print("✅ Feature dataset saved to data/features_dataset.parquet")
