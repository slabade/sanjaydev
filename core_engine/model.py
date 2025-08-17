
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/xgb_model.json")

FEATURE_COLS = ["last_price","iv","delta_scaled","days_to_expiry","moneyness"]

def make_labels(df: pd.DataFrame, up_thresh=0.20) -> pd.Series:
    """Label = 1 if max_price_until_expiry >= last_price * (1+up_thresh)."""
    return (df["max_price_until_expiry"] >= df["last_price"] * (1.0 + up_thresh)).astype(int)

def train_model(features_path="data/features_dataset.parquet"):
    df = pd.read_parquet(features_path)
    y = make_labels(df, up_thresh=0.20)
    X = df[FEATURE_COLS]

    # TimeSeries split on as_of_date ordering
    order = np.argsort(pd.to_datetime(df["as_of_date"]).values)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    final_model = None
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            tree_method="hist"
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict_proba(X.iloc[te])[:,1]
        auc = roc_auc_score(y.iloc[te], preds)
        aucs.append(auc)
        print(f"Fold {fold} AUC: {auc:.4f}")
        if fold == 5:
            final_model = model

    print(f"Mean AUC: {np.mean(aucs):.4f}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_PATH.as_posix())
    print(f"✅ Model saved to {MODEL_PATH}")

def predict_today(features_path="data/features_dataset.parquet", out_csv="data/recommended_trades.csv", top_n=20):
    df = pd.read_parquet(features_path).copy()
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH.as_posix())

    X = df[FEATURE_COLS]
    df["pred_prob"] = model.predict_proba(X)[:,1]

    # Use the latest date in the dataset as "today"
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.tz_localize(None)
    latest_date = df["as_of_date"].max()
    today_df = df[df["as_of_date"] == latest_date]
    recs = today_df.sort_values("pred_prob", ascending=False).head(top_n)
    recs.to_csv(out_csv, index=False)
    print(f"✅ Today's recommendations saved to {out_csv} (date={latest_date.date()})")
