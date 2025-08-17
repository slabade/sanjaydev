
import pandas as pd
from datetime import datetime

class ExecutionSimulator:
    """Week 6–8: simple execution simulator with slippage & commission."""
    def __init__(self, starting_cash=100000.0, commission_per_trade=1.0, slippage_pct=0.002):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.commission = float(commission_per_trade)
        self.slippage_pct = float(slippage_pct)
        self.positions = []
        self.history = []
        self.pnl_history = []

    def _apply_slippage(self, price, side="buy"):
        return price * (1 + self.slippage_pct) if side == "buy" else price * (1 - self.slippage_pct)

    def open_position(self, trade: dict, qty_contracts: int):
        entry_price = float(trade.get("last_price", 0.0))
        if entry_price <= 0:
            return False, "Invalid price"
        entry_price_slipped = self._apply_slippage(entry_price, "buy")
        notional = entry_price_slipped * qty_contracts * 100.0
        total_cost = notional + self.commission
        if total_cost > self.cash:
            return False, "Insufficient cash"
        pos_id = f"pos_{len(self.positions)+1}_{int(datetime.utcnow().timestamp()*1000)}"
        pos = {
            "id": pos_id,
            "symbol": trade.get("symbol"),
            "qty": int(qty_contracts),
            "entry_price": entry_price_slipped,
            "expiry": trade.get("expiry"),
            "as_of_date": trade.get("as_of_date"),
            "max_price_until_expiry": trade.get("max_price_until_expiry", entry_price_slipped),
        }
        self.positions.append(pos)
        self.cash -= total_cost
        self.history.append({"time": datetime.utcnow().isoformat(), "action": "buy", "pos_id": pos_id,
                             "price": entry_price_slipped, "qty": qty_contracts, "cost": total_cost})
        return True, pos_id

    def close_position(self, pos, exit_price=None):
        px = exit_price if exit_price is not None else pos.get("max_price_until_expiry", pos["entry_price"])
        exit_price_slipped = self._apply_slippage(float(px), "sell")
        revenue = exit_price_slipped * pos["qty"] * 100.0 - self.commission
        self.cash += revenue
        pnl = revenue - (pos["entry_price"] * pos["qty"] * 100.0 + self.commission)
        self.history.append({
            "time": datetime.utcnow().isoformat(), "action": "sell", "pos_id": pos["id"],
            "price": exit_price_slipped, "qty": pos["qty"], "revenue": revenue, "pnl": pnl
        })
        self.positions = [p for p in self.positions if p["id"] != pos["id"]]
        return pnl

    def liquidate_all(self):
        for p in list(self.positions):
            self.close_position(p, exit_price=p.get("max_price_until_expiry", p["entry_price"]))

    def snapshot(self, time_label=None):
        pv = self.cash
        for p in self.positions:
            mark = p.get("max_price_until_expiry", p["entry_price"])
            pv += mark * p["qty"] * 100.0
        snap = {"time": time_label or datetime.utcnow().isoformat(),
                "cash": self.cash, "positions": len(self.positions), "portfolio_value": pv,
                "starting_cash": self.starting_cash}
        self.pnl_history.append(snap)
        return snap

    def export_snapshot_df(self):
        return pd.DataFrame(self.pnl_history) if self.pnl_history else pd.DataFrame([self.snapshot()])
