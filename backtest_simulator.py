
import os
import pandas as pd
from core_engine.execution import ExecutionSimulator
from datetime import datetime

TRAINING_FILE = "data/training_dataset.parquet"
RECOMMEND_CSV = "data/recommended_trades.csv"
OUT_SIM_PARQUET = "data/sim_results.parquet"
OUT_HISTORY_CSV = "data/sim_history.csv"

DAILY_BUDGET_PCT = 0.02
MAX_CONTRACTS_PER_TRADE = 5
ENTRY_FEE = 1.0
SLIPPAGE = 0.002

def run_backtest_from_recommendations():
    if os.path.exists(RECOMMEND_CSV):
        recs = pd.read_csv(RECOMMEND_CSV)
    else:
        recs = pd.read_parquet(TRAINING_FILE).sample(frac=0.01, random_state=42).reset_index(drop=True)

    sim = ExecutionSimulator(starting_cash=100000.0, commission_per_trade=ENTRY_FEE, slippage_pct=SLIPPAGE)

    for _, row in recs.iterrows():
        budget = sim.starting_cash * DAILY_BUDGET_PCT
        entry_price = float(row.get("last_price", 0.0))
        if entry_price <= 0:
            continue
        qty = int(min(MAX_CONTRACTS_PER_TRADE, max(1, budget // (entry_price * 100))))
        ok, _ = sim.open_position(row.to_dict(), qty)
        if not ok and qty > 1:
            sim.open_position(row.to_dict(), qty-1)
        sim.snapshot(time_label=row.get("as_of_date", datetime.utcnow().isoformat()))

    sim.liquidate_all()
    sim.snapshot(time_label="final")
    snaps = sim.export_snapshot_df()
    snaps.to_parquet(OUT_SIM_PARQUET, index=False)
    pd.DataFrame(sim.history).to_csv(OUT_HISTORY_CSV, index=False)
    print(f"✅ Backtest complete. Snapshots → {OUT_SIM_PARQUET}; History → {OUT_HISTORY_CSV}")

if __name__ == "__main__":
    run_backtest_from_recommendations()
