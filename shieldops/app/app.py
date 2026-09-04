"""
ShieldOps — Flask Backend
===========================
Routes:
  GET  /                 -> redirects to dashboard
  GET  /dashboard         -> renders the dashboard (flagged cases + metrics)
  POST /run-batch         -> re-scores the full synthetic dataset (the "batch")
  GET  /audit/<id>        -> full audit detail for one flagged case
  POST /demo-failure       -> forces one Gemini call to fail, to DEMO the
                              graceful fallback live (for the pitch video)

Run:  python3 app/app.py
Then open http://localhost:5000
"""

import os
import sys
import sqlite3
import datetime as dt

import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request, redirect

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import (
    build_customer_features, build_order_features,
    FEATURE_COLS_RETURN_ABUSE, FEATURE_COLS_CHARGEBACK,
)
from agent import llm_agent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "saved")
DB_PATH = os.path.join(BASE_DIR, "app", "audit_trail.db")

app = Flask(__name__)

return_model = joblib.load(os.path.join(MODEL_DIR, "return_abuse_model.pkl"))
chargeback_model = joblib.load(os.path.join(MODEL_DIR, "chargeback_model.pkl"))

RETURN_ABUSE_THRESHOLD = 0.30   # flag for review above this score
# Training explored 0.30 (higher recall, more false positives, see ml/train_models.py).
# For the live dashboard we use 0.50 to keep the flagged queue reviewable —
# this tradeoff (and both sets of numbers) is documented in the README.
CHARGEBACK_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            entity_id TEXT,
            risk_score REAL,
            risk_tier TEXT,
            explanation TEXT,
            source TEXT,
            recommended_action TEXT,
            estimated_cost_impact REAL,
            created_at TEXT
        )
    """)
    con.commit()
    con.close()


def risk_tier(score):
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Batch scoring — this is the core "agent" workflow
# ---------------------------------------------------------------------------
def run_batch(force_llm_failure=False):
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))

    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM audit_log")  # fresh batch each run, for a clean demo
    cur = con.cursor()
    now = dt.datetime.now().isoformat(timespec="seconds")

    old_key = llm_agent.GEMINI_API_KEY
    if force_llm_failure:
        llm_agent.GEMINI_API_KEY = "FORCE_INVALID_KEY_FOR_DEMO"

    # ---- Return abuse batch ----
    cust_feats = build_customer_features(customers, orders)
    X = cust_feats[FEATURE_COLS_RETURN_ABUSE].fillna(0)
    scores = return_model.predict_proba(X)[:, 1]
    cust_feats["risk_score"] = scores

    flagged_customers = cust_feats[cust_feats["risk_score"] >= RETURN_ABUSE_THRESHOLD]
    for _, row in flagged_customers.iterrows():
        result = llm_agent.explain_return_flag(row.to_dict())
        action = "Hold future high-value orders for manual review" if row["risk_score"] >= 0.6 else "Monitor next 2 orders"
        cost = 4500 if row["risk_score"] >= 0.6 else 1500
        cur.execute(
            "INSERT INTO audit_log (entity_type, entity_id, risk_score, risk_tier, "
            "explanation, source, recommended_action, estimated_cost_impact, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("return_abuse", row["customer_id"], float(row["risk_score"]),
             risk_tier(row["risk_score"]), result["explanation"], result["source"],
             action, cost, now),
        )

    # ---- Chargeback batch ----
    order_feats = build_order_features(orders, customers)
    Xo = order_feats[FEATURE_COLS_CHARGEBACK].fillna(0)
    oscores = chargeback_model.predict_proba(Xo)[:, 1]
    order_feats["risk_score"] = oscores

    flagged_orders = order_feats[order_feats["risk_score"] >= CHARGEBACK_THRESHOLD]
    for _, row in flagged_orders.iterrows():
        result = llm_agent.generate_evidence_draft(row.to_dict())
        action = "Evidence packet pre-drafted for dispute defense" if row["delivery_confirmed"] else "Escalate: delivery unconfirmed, verify courier POD urgently"
        cost = float(row["order_value"])
        cur.execute(
            "INSERT INTO audit_log (entity_type, entity_id, risk_score, risk_tier, "
            "explanation, source, recommended_action, estimated_cost_impact, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("chargeback", row["order_id"], float(row["risk_score"]),
             risk_tier(row["risk_score"]), result["evidence_draft"], result["source"],
             action, cost, now),
        )

    con.commit()
    con.close()
    llm_agent.GEMINI_API_KEY = old_key

    return {
        "return_abuse_flagged": len(flagged_customers),
        "chargeback_flagged": len(flagged_orders),
        "total_customers_scanned": len(cust_feats),
        "total_orders_scanned": len(order_feats),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM audit_log ORDER BY risk_score DESC").fetchall()
    con.close()

    rows = [dict(r) for r in rows]
    total_flagged = len(rows)
    total_cost_at_risk = sum(r["estimated_cost_impact"] for r in rows)
    llm_count = sum(1 for r in rows if r["source"] == "llm_generated")
    fallback_count = sum(1 for r in rows if r["source"] == "template_fallback")

    return render_template(
        "dashboard.html",
        rows=rows,
        total_flagged=total_flagged,
        total_cost_at_risk=total_cost_at_risk,
        llm_count=llm_count,
        fallback_count=fallback_count,
    )


@app.route("/run-batch", methods=["POST"])
def run_batch_route():
    stats = run_batch(force_llm_failure=False)
    return jsonify(stats)


@app.route("/demo-failure", methods=["POST"])
def demo_failure_route():
    """Forces every LLM call in this batch to fail -> shows the graceful
    template fallback kicking in. Use this live in the pitch video."""
    stats = run_batch(force_llm_failure=True)
    return jsonify({**stats, "note": "LLM calls forced to fail — check 'source' column, all should say template_fallback"})


if __name__ == "__main__":
    init_db()
    print("Running initial batch scan...")
    stats = run_batch()
    print(stats)
    app.run(debug=True, port=5000)
