"""
ShieldOps — LLM Agent Layer (Gemini)
======================================
Turns ML model output into human-readable explanations and evidence drafts.

IMPORTANT DESIGN CHOICE (this is your "failure handled gracefully" story
for the pitch video):
  Every Gemini call is wrapped in try/except with a timeout. If the API key
  is missing, the network fails, the quota is exhausted, or the response is
  malformed -> we NEVER crash and NEVER block the merchant's workflow.
  Instead we fall back to a deterministic, template-based explanation built
  straight from the feature values. The audit trail logs WHICH path was used
  (llm_generated vs template_fallback) so it's fully transparent.
"""

import os
import requests


def _load_api_key():
    """Load API key from environment variable or search in .env files."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # Search for .env files in likely directories
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            parsed_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if parsed_key:
                                return parsed_key
            except Exception:
                pass
    return ""


GEMINI_API_KEY = _load_api_key()
GEMINI_MODEL = "gemini-3.1-flash-lite"
TIMEOUT_SECONDS = 15


def _call_gemini(prompt: str) -> str | None:
    """Returns generated text, or None if anything goes wrong (never raises)."""
    # Use dynamically in case GEMINI_API_KEY was overridden or loaded late
    active_key = GEMINI_API_KEY or _load_api_key()
    if not active_key or active_key == "FORCE_INVALID_KEY_FOR_DEMO":
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={active_key}"
    )

    try:
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        print(f"[llm_agent] Gemini call failed, using fallback. Reason: {e}")
        return None


# ---------------------------------------------------------------------------
# Use case 1: Explain a RETURN ABUSE flag
# ---------------------------------------------------------------------------
def explain_return_flag(customer_row: dict) -> dict:
    prompt = f"""You are a fraud-risk analyst assistant for an e-commerce merchant.
Explain in 2 short sentences, in plain English, why this customer was flagged
for possible return abuse. Be specific, cite the numbers given, and stay
factual and non-accusatory — this is a risk signal for human review, not a
final verdict.

Customer data:
- Total orders: {customer_row.get('total_orders')}
- Total returns: {customer_row.get('total_returns')}
- Return rate: {customer_row.get('return_rate', 0):.0%}
- Avg days taken to return an item: {customer_row.get('avg_days_to_return', 0):.0f}
- High-value item return ratio: {customer_row.get('high_value_return_ratio', 0):.0%}
- Shares delivery address with another account: {'Yes' if customer_row.get('shares_address') else 'No'}
"""
    text = _call_gemini(prompt)
    if text:
        return {"explanation": text, "source": "llm_generated"}

    # ---- Graceful fallback: deterministic template from the same features ----
    reasons = []
    if customer_row.get("return_rate", 0) >= 0.4:
        reasons.append(f"a high return rate of {customer_row['return_rate']:.0%}")
    if customer_row.get("high_value_return_ratio", 0) >= 0.3:
        reasons.append("a pattern of returning mostly high-value items")
    if customer_row.get("shares_address"):
        reasons.append("sharing a delivery address with another flagged account")
    if customer_row.get("avg_days_to_return", 0) >= 20:
        reasons.append("returns filed close to the policy deadline (wardrobing pattern)")
    if not reasons:
        reasons.append("a combination of moderate risk signals")

    template = f"Flagged for review due to {', '.join(reasons)}. Recommend manual review before approving future high-value orders."
    return {"explanation": template, "source": "template_fallback"}


# ---------------------------------------------------------------------------
# Use case 2: Draft a CHARGEBACK evidence packet
# ---------------------------------------------------------------------------
def generate_evidence_draft(order_row: dict) -> dict:
    prompt = f"""You are helping a merchant prepare a chargeback dispute response
for a payment gateway. Write a short, professional evidence summary (3-4
sentences) a merchant could submit, based ONLY on the facts given below.
Do not invent any facts not listed.

Order facts:
- Order ID: {order_row.get('order_id')}
- Order value: Rs.{order_row.get('order_value')}
- Delivery confirmed: {'Yes' if order_row.get('delivery_confirmed') else 'No'}
- Payment method: {order_row.get('payment_method')}
- Customer's prior orders with this merchant: {order_row.get('customer_prior_orders')}
- Customer's prior chargebacks: {order_row.get('customer_prior_chargebacks')}
- Predicted chargeback risk score: {order_row.get('risk_score', 0):.0%}
"""
    text = _call_gemini(prompt)
    if text:
        return {"evidence_draft": text, "source": "llm_generated"}

    # ---- Graceful fallback template ----
    delivery_line = (
        "Delivery was confirmed for this order."
        if order_row.get("delivery_confirmed")
        else "WARNING: delivery was NOT confirmed for this order — verify courier POD before submitting."
    )
    template = (
        f"Evidence summary for order {order_row.get('order_id')}: "
        f"Order value Rs.{order_row.get('order_value')} paid via {order_row.get('payment_method')}. "
        f"{delivery_line} "
        f"Customer has {order_row.get('customer_prior_orders')} prior orders and "
        f"{order_row.get('customer_prior_chargebacks')} prior chargebacks with this merchant. "
        f"Model-predicted risk score at time of order: {order_row.get('risk_score', 0):.0%}."
    )
    return {"evidence_draft": template, "source": "template_fallback"}
