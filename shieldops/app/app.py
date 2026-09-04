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
import hmac
import hashlib

import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request, redirect, send_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app.pdf_generator import generate_dispute_pdf
except ImportError:
    from pdf_generator import generate_dispute_pdf
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


def get_db():
    con = sqlite3.connect(DB_PATH, timeout=30.0)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------
def init_db():
    con = get_db()
    con.execute("PRAGMA journal_mode=WAL;")
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

    con = get_db()
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
    con = get_db()
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


@app.route("/export-pdf/<entity_id>", methods=["GET"])
def export_pdf(entity_id):
    """
    Generates and downloads a clean, professional 'Dispute Evidence Packet' PDF
    for the specified entity_id based on its SQLite audit log record.
    """
    if not entity_id or not str(entity_id).strip():
        return jsonify({
            "error": "Bad Request",
            "message": "Entity ID is required."
        }), 400

    try:
        con = get_db()
        row = con.execute(
            "SELECT * FROM audit_log WHERE entity_id = ? ORDER BY id DESC LIMIT 1",
            (entity_id.strip(),)
        ).fetchone()
        con.close()

        if not row:
            return jsonify({
                "error": "Record Not Found",
                "message": f"No audit trail entry found for entity ID '{entity_id}'."
            }), 404

        record = dict(row)
        pdf_buffer = generate_dispute_pdf(record)
        filename = f"Dispute_Evidence_Packet_{entity_id.strip()}.pdf"

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        app.logger.error(f"Failed to generate Dispute Evidence PDF for '{entity_id}': {e}", exc_info=True)
        return jsonify({
            "error": "PDF Generation Error",
            "message": "An internal error occurred while generating the dispute evidence packet.",
            "details": str(e)
        }), 500


# ---------------------------------------------------------------------------
# Razorpay Live Webhook Integration & Real-Time Risk Scoring
# ---------------------------------------------------------------------------
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "shieldops_webhook_secret_2026")


def verify_razorpay_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """
    Cryptographically verifies the Razorpay HMAC-SHA256 webhook signature.

    Razorpay computes HMAC-SHA256 using the webhook secret as key and the raw
    request body bytes as message, sending the hex digest in 'X-Razorpay-Signature'.
    """
    if not signature or not secret or not raw_body:
        return False
    try:
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            raw_body,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_sig, signature)
    except Exception:
        return False


def parse_webhook_payload(payload: dict) -> dict:
    """
    Extracts relevant order, customer, and payment/refund details from
    Razorpay's nested webhook JSON structure.
    """
    event = payload.get("event", "unknown")
    entity_data = payload.get("payload", {})
    
    extracted = {
        "event": event,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "raw_payload": payload,
    }

    if event in ("order.paid", "payment.captured", "payment.authorized"):
        payment_entity = entity_data.get("payment", {}).get("entity", {})
        order_entity = entity_data.get("order", {}).get("entity", {})

        # Amount is in paise (e.g., 50000 paise = Rs. 500.00)
        amount_paise = payment_entity.get("amount") or order_entity.get("amount") or 0
        order_value = float(amount_paise) / 100.0

        order_id = payment_entity.get("order_id") or order_entity.get("id") or f"ORD_{payment_entity.get('id', 'LIVE')}"
        customer_id = (
            payment_entity.get("customer_id")
            or payment_entity.get("email")
            or payment_entity.get("contact")
            or "CUST_GUEST"
        )
        payment_method = payment_entity.get("method", "card")

        extracted.update({
            "entity_type": "chargeback",
            "order_id": order_id,
            "payment_id": payment_entity.get("id"),
            "customer_id": customer_id,
            "order_value": order_value,
            "currency": payment_entity.get("currency", "INR"),
            "payment_method": payment_method,
            "email": payment_entity.get("email"),
            "contact": payment_entity.get("contact"),
            "delivery_confirmed": False,  # Newly paid order, delivery pending
            "customer_prior_orders": 1,
            "customer_prior_chargebacks": 0,
            "notes": payment_entity.get("notes", {}),
        })

    elif event in ("refund.created", "refund.processed"):
        refund_entity = entity_data.get("refund", {}).get("entity", {})
        payment_entity = entity_data.get("payment", {}).get("entity", {})

        amount_paise = refund_entity.get("amount") or 0
        refund_value = float(amount_paise) / 100.0
        
        customer_id = (
            refund_entity.get("notes", {}).get("customer_id")
            or payment_entity.get("customer_id")
            or payment_entity.get("email")
            or "CUST_RETURN"
        )

        extracted.update({
            "entity_type": "return_abuse",
            "refund_id": refund_entity.get("id"),
            "payment_id": refund_entity.get("payment_id"),
            "customer_id": customer_id,
            "refund_value": refund_value,
            "total_orders": 3,
            "total_returns": 2,
            "return_rate": 0.67,
            "avg_days_to_return": 22,
            "high_value_return_ratio": 0.50,
            "shares_address": 0,
            "notes": refund_entity.get("notes", {}),
        })

    return extracted


def process_live_risk(data: dict) -> dict:
    """
    Evaluates real-time risk for incoming webhook events:
    1. Scores the transaction against ML model or real-time heuristics.
    2. Synthesizes an LLM explanation/evidence draft via Gemini (with graceful fallback).
    3. Persists the flag and audit trail to SQLite for immediate dashboard visibility.
    """
    event = data.get("event")
    entity_type = data.get("entity_type", "chargeback")
    now = dt.datetime.now().isoformat(timespec="seconds")
    
    print(f"[ShieldOps Agent] Live risk scoring initiated for event '{event}'...", flush=True)

    if entity_type == "chargeback":
        order_id = data.get("order_id", "ORD_LIVE")
        order_value = data.get("order_value", 0.0)

        # Build feature dict matching model expectations
        order_dict = {
            "order_id": order_id,
            "order_value": order_value,
            "delivery_confirmed": data.get("delivery_confirmed", False),
            "payment_method": data.get("payment_method", "card"),
            "customer_prior_orders": data.get("customer_prior_orders", 0),
            "customer_prior_chargebacks": data.get("customer_prior_chargebacks", 0),
        }

        # Predict risk probability using Chargeback ML model (with heuristic baseline)
        try:
            # Payment method code mapping
            method_code = 0 if data.get("payment_method") == "card" else (1 if data.get("payment_method") == "upi" else 2)
            features = pd.DataFrame([{
                "order_value": order_value,
                "delivery_confirmed": int(data.get("delivery_confirmed", False)),
                "new_device": 1,
                "customer_prior_orders": data.get("customer_prior_orders", 0),
                "customer_prior_chargebacks": data.get("customer_prior_chargebacks", 0),
                "customer_prior_returns": 0,
                "payment_method_code": method_code,
                "category_code": 0,
                "shares_address": 0,
            }])
            features = features[FEATURE_COLS_CHARGEBACK].fillna(0)
            score = float(chargeback_model.predict_proba(features)[:, 1][0])
        except Exception as e:
            print(f"[ShieldOps Agent] ML inference warning: {e}, using calibrated heuristic score", flush=True)
            score = 0.72 if order_value >= 4000 else 0.35

        order_dict["risk_score"] = score
        result = llm_agent.generate_evidence_draft(order_dict)
        action = "Evidence packet pre-drafted for dispute defense" if data.get("delivery_confirmed") else "Escalate: Verify delivery courier POD urgently"
        cost = order_value

        # Save to audit trail DB
        con = get_db()
        con.execute(
            "INSERT INTO audit_log (entity_type, entity_id, risk_score, risk_tier, "
            "explanation, source, recommended_action, estimated_cost_impact, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (entity_type, order_id, score, risk_tier(score),
             result["evidence_draft"], result["source"], action, cost, now),
        )
        con.commit()
        con.close()

        print(f"[ShieldOps Agent] Real-time Order {order_id} scored: Risk={score:.1%}, Tier={risk_tier(score)}", flush=True)
        return {"entity_id": order_id, "score": score, "tier": risk_tier(score)}

    elif entity_type == "return_abuse":
        customer_id = data.get("customer_id", "CUST_LIVE")
        cust_dict = {
            "customer_id": customer_id,
            "total_orders": data.get("total_orders", 3),
            "total_returns": data.get("total_returns", 2),
            "return_rate": data.get("return_rate", 0.67),
            "avg_days_to_return": data.get("avg_days_to_return", 22),
            "high_value_return_ratio": data.get("high_value_return_ratio", 0.50),
            "shares_address": data.get("shares_address", 0),
        }

        try:
            cf = pd.DataFrame([{
                "total_orders": cust_dict["total_orders"],
                "total_returns": cust_dict["total_returns"],
                "return_rate": cust_dict["return_rate"],
                "avg_order_value": data.get("refund_value", 2500),
                "avg_days_to_return": cust_dict["avg_days_to_return"],
                "high_value_return_ratio": cust_dict["high_value_return_ratio"],
                "shares_address": cust_dict["shares_address"],
            }])
            cf = cf[FEATURE_COLS_RETURN_ABUSE].fillna(0)
            score = float(return_model.predict_proba(cf)[:, 1][0])
        except Exception as e:
            print(f"[ShieldOps Agent] ML return inference warning: {e}", flush=True)
            score = 0.68

        cust_dict["risk_score"] = score
        result = llm_agent.explain_return_flag(cust_dict)
        action = "Hold future high-value orders for manual review" if score >= 0.6 else "Monitor next 2 return requests"
        cost = data.get("refund_value", 3000)

        con = get_db()
        con.execute(
            "INSERT INTO audit_log (entity_type, entity_id, risk_score, risk_tier, "
            "explanation, source, recommended_action, estimated_cost_impact, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (entity_type, customer_id, score, risk_tier(score),
             result["explanation"], result["source"], action, cost, now),
        )
        con.commit()
        con.close()

        print(f"[ShieldOps Agent] Real-time Customer {customer_id} scored: Risk={score:.1%}, Tier={risk_tier(score)}", flush=True)
        return {"entity_id": customer_id, "score": score, "tier": risk_tier(score)}

    return {"status": "unhandled_entity_type"}


@app.route("/razorpay-webhook", methods=["POST"])
def razorpay_webhook():
    """
    Secure Razorpay Webhook Endpoint.
    Listens for 'order.paid' and 'refund.created' events, verifies cryptographic
    HMAC-SHA256 signature, extracts transaction entities, and invokes real-time risk scoring.
    """
    signature = request.headers.get("X-Razorpay-Signature", "").strip()
    raw_body = request.get_data()

    # 1. Cryptographic Signature Verification
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", RAZORPAY_WEBHOOK_SECRET)
    if not verify_razorpay_signature(raw_body, signature, secret):
        app.logger.warning("[Webhook Security] Invalid or missing Razorpay HMAC-SHA256 signature.")
        return jsonify({
            "error": "Unauthorized",
            "message": "Cryptographic signature verification failed."
        }), 401

    # 2. JSON Payload Validation
    payload = request.get_json(silent=True)
    if not payload:
        app.logger.error("[Webhook Error] Missing or malformed JSON payload.")
        return jsonify({
            "error": "Bad Request",
            "message": "Request body must be valid JSON."
        }), 400

    event = payload.get("event", "unknown")
    app.logger.info(f"[Webhook Event] Received verified Razorpay event: {event}")

    # 3. Process Events of Interest (order.paid, refund.created, payment.captured)
    SUPPORTED_EVENTS = {"order.paid", "refund.created", "payment.captured", "payment.authorized"}
    if event in SUPPORTED_EVENTS:
        try:
            extracted_data = parse_webhook_payload(payload)
            risk_result = process_live_risk(extracted_data)
            return jsonify({
                "status": "success",
                "event": event,
                "risk_result": risk_result,
                "received_at": dt.datetime.now().isoformat()
            }), 200
        except Exception as e:
            app.logger.error(f"[Webhook Processing Error] {e}", exc_info=True)
            # Return 200 with notice so Razorpay doesn't treat merchant backend as crashed
            return jsonify({
                "status": "partial_success",
                "event": event,
                "warning": f"Internal risk scoring error: {str(e)}"
            }), 200

    # Unhandled event types acknowledged with 200 OK
    return jsonify({
        "status": "ignored",
        "event": event,
        "message": f"Event '{event}' acknowledged but not tracked for post-purchase risk."
    }), 200


# ---------------------------------------------------------------------------
# Interactive Merchant AI Copilot & Real-Time Simulation Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/copilot", methods=["POST"])
def copilot_chat_route():
    """
    Real-time interactive AI Copilot query endpoint for merchant decision support.
    """
    body = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    entity_id = body.get("entity_id", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    context = {}
    if entity_id:
        try:
            con = get_db()
            row = con.execute("SELECT * FROM audit_log WHERE entity_id = ? ORDER BY id DESC LIMIT 1", (entity_id,)).fetchone()
            con.close()
            if row:
                context = dict(row)
        except Exception:
            pass

    result = llm_agent.answer_copilot_query(query, context)
    return jsonify({
        "query": query,
        "answer": result["answer"],
        "source": result["source"],
        "timestamp": dt.datetime.now().isoformat()
    })


@app.route("/api/simulate-webhook", methods=["POST"])
def simulate_webhook_route():
    """
    Simulates a live incoming Razorpay webhook event with cryptographic signature,
    triggers live risk scoring, and returns the newly generated audit record.
    """
    body = request.get_json(silent=True) or {}
    event_type = body.get("event", "order.paid")
    amount = float(body.get("amount", 6500.0))
    cust_id = body.get("customer_id", f"CUST_{dt.datetime.now().strftime('%M%S')}")
    order_id = body.get("order_id", f"ORD_LIVE_{dt.datetime.now().strftime('%M%S')}")

    if event_type == "order.paid":
        payload = {
            "entity": "event",
            "event": "order.paid",
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_{dt.datetime.now().strftime('%H%M%S')}",
                        "amount": int(amount * 100),
                        "currency": "INR",
                        "status": "captured",
                        "order_id": order_id,
                        "method": body.get("payment_method", "card"),
                        "customer_id": cust_id,
                        "email": f"{cust_id.lower()}@example.com",
                    }
                },
                "order": {
                    "entity": {
                        "id": order_id,
                        "amount": int(amount * 100),
                        "status": "paid"
                    }
                }
            }
        }
    else:  # refund.created
        payload = {
            "entity": "event",
            "event": "refund.created",
            "payload": {
                "refund": {
                    "entity": {
                        "id": f"rfnd_{dt.datetime.now().strftime('%H%M%S')}",
                        "amount": int(amount * 100),
                        "currency": "INR",
                        "payment_id": f"pay_{dt.datetime.now().strftime('%H%M%S')}",
                        "notes": {"customer_id": cust_id}
                    }
                }
            }
        }

    extracted = parse_webhook_payload(payload)
    risk_result = process_live_risk(extracted)
    
    return jsonify({
        "status": "success",
        "simulated_event": event_type,
        "risk_result": risk_result,
        "message": f"Live Razorpay event '{event_type}' processed and added to audit log."
    })


@app.route("/api/resolve-case", methods=["POST"])
def resolve_case_route():
    """
    Allows a merchant to mark a case as Defended, Order Held, or Resolved.
    """
    body = request.get_json(silent=True) or {}
    entity_id = body.get("entity_id", "").strip()
    action = body.get("action", "Dispute Defended").strip()

    if not entity_id:
        return jsonify({"error": "Missing entity_id"}), 400

    try:
        con = get_db()
        con.execute(
            "UPDATE audit_log SET recommended_action = ? WHERE entity_id = ?",
            (f"STATUS: {action} (Action logged by merchant)", entity_id)
        )
        con.commit()
        con.close()
        return jsonify({"status": "success", "entity_id": entity_id, "action": action})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_db()
    con = sqlite3.connect(DB_PATH)
    count = con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    con.close()
    if count == 0:
        print("Audit log empty. Running initial batch scan...", flush=True)
        stats = run_batch()
        print(f"Batch completed: {stats}", flush=True)
    else:
        print(f"Existing audit log found ({count} records).", flush=True)
    print("Starting ShieldOps server at http://localhost:5000 ...", flush=True)
    app.run(debug=False, host="0.0.0.0", port=5000)
