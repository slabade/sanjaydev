
import os, io
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from json2html import *

app = FastAPI(title="AI Options Recommender Dashboard")


RECOMMEND_PATH = "data/recommended_trades.csv"
SIM_PARQUET = "data/sim_results.parquet"

@app.get("/", response_class=HTMLResponse)
def home():
    html = "<h2>AI Options Recommender — Dashboard</h2>"
    html += "<p><a href='/'>/Home</a> <a href='/alerts'>/alerts</a> | <a href='/metrics'>/metrics</a></p>"
    if os.path.exists(SIM_PARQUET):
        html += "<h3>Portfolio Curve</h3><img src='/plot/curve.png'/>"
    else:
        html += "<p>No simulation results yet. Run <code>python backtest_simulator.py</code>.</p>"
    return HTMLResponse(content=html)

@app.get("/alerts")
def alerts():
    html = "<h2>AI Options Recommender — Dashboard</h2>"
    html += "<p><a href='/'>/Home</a> <a href='/alerts'>/alerts</a> | <a href='/metrics'>/metrics</a></p>"

    if not os.path.exists(RECOMMEND_PATH):
        return {"count": 0, "alerts": []}
    df = pd.read_csv(RECOMMEND_PATH)
    output = json2html.convert(json=df.to_dict(orient="records"))
    html += output
    return HTMLResponse(content=html)

@app.get("/metrics")
def metrics():
    html = "<h2>AI Options Recommender — Dashboard</h2>"
    html += "<p><a href='/'>/Home</a> <a href='/alerts'>/alerts</a> | <a href='/metrics'>/metrics</a></p>"
    if not os.path.exists(SIM_PARQUET):
        return {"message": "No simulation results found."}
    df = pd.read_parquet(SIM_PARQUET)
    latest = df.iloc[-1].to_dict() if not df.empty else {}
    output = json2html.convert(json=latest)
    html += output
    return  HTMLResponse(content=html) #{"snapshots_count": len(df), "latest_snapshot": latest}

@app.get("/plot/curve.png")
def plot_curve():
    if not os.path.exists(SIM_PARQUET):
        return Response(status_code=404, content=b"")
    df = pd.read_parquet(SIM_PARQUET)
    plt.figure(figsize=(8,4))
    plt.plot(df["portfolio_value"].astype(float), marker=".", linestyle="-")
    plt.title("Portfolio Value (sim)")
    plt.xlabel("Snapshot")
    plt.ylabel("Portfolio Value")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
