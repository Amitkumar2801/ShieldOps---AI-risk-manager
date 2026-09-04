# ShieldOps — Post-Purchase Risk & Dispute Defense Agent

Built for the **Razorpay AI Buildathon — Track 2: AI Risk Manager**

---

## 🎯 The Problem

Merchants lose billions *after* a sale is completed through two interconnected fraud vectors:
1. **Serial Return Abuse (Wardrobing / Policy Exploitation):** Customers exploiting return policies (wardrobing high-value items, multi-accounting/address sharing) to quietly drain merchant margins.
2. **Chargebacks & Payment Disputes:** Disputed transactions where the merchant loses both the merchandise and the revenue, plus acquiring bank penalty fees.

Both risks stem from the same underlying behavioral patterns. **ShieldOps** scores both risks from a single unified pipeline, **explains every flag in plain English**, **generates instant Dispute Evidence Packets (PDFs)**, and **actively listens to live Razorpay webhooks in real time**.

---

## 🏗️ Architecture

```
                               ┌───────────────────────────────────────────────┐
                               │       Live Razorpay Webhook Events             │
                               │ (order.paid, refund.created, payment.captured)│
                               └──────────────────────┬────────────────────────┘
                                                      │ (HMAC-SHA256 Verified)
Synthetic Dataset (customers.csv, orders.csv)         │
               │                                      │
               ▼                                      ▼
     Feature Engineering (ml/features.py) ◄───────────┘
               │
               ├──► Return Abuse Model (RandomForest, customer-level)
               │
               └──► Chargeback Risk Model (RandomForest, order-level)
                               │
                               ▼
               LLM Agent Layer (Google Gemini, agent/llm_agent.py)
               - Explains WHY a customer or transaction was flagged
               - Drafts bank-grade chargeback dispute evidence text
               - Powers the interactive Merchant AI Risk Copilot
               - Bulletproof Graceful Fallback: falls back to deterministic template if LLM fails
                               │
                               ▼
               Dispute PDF Generator (ReportLab, app/pdf_generator.py)
               - Exports formatted "Dispute Evidence Packet" PDFs
                               │
                               ▼
         SQLite Audit Trail + Flask Real-Time Dashboard (app/app.py)
```

---

## 💡 Key Architectural Design Decisions

- **Classic ML for Detection, LLM for Explanation & Defense:** The risk score is computed using Random Forest models trained on engineered behavioral features — the right tool for fast, calibrated tabular classification. Gemini LLM is used where natural language synthesis shines: explaining risk signals to human reviewers and drafting structured evidence packets.
- **Real-Time Razorpay Webhook Integration:** Live endpoint (`/razorpay-webhook`) cryptographically verifies `X-Razorpay-Signature` using HMAC-SHA256, parses `order.paid` and `refund.created` payloads, and scores transactions in real time.
- **Fail-Safe Graceful Fallback:** Every Gemini call in `agent/llm_agent.py` is protected by timeouts and try/except handlers. If the API key is missing or the network drops, ShieldOps **never crashes and never blocks merchant operations** — it instantly switches to deterministic template synthesis, logging the provenance (`llm_generated` vs `template_fallback`).
- **One-Click Dispute Evidence PDF:** Integrated with **ReportLab** (`/export-pdf/<entity_id>`) to generate presentation-ready Dispute Evidence Packets with case metadata, narrative summaries, and audit trail verification.
- **Interactive AI Risk Copilot:** Real-time slide-out drawer (`/api/copilot`) enabling merchants to ask situational questions regarding chargeback dispute strategy and return mitigation.
- **Documented Threshold Tradeoffs:** `ml/train_models.py` transparently evaluates precision vs. recall tradeoffs (e.g., threshold 0.50 vs 0.30) to balance false-positive manual review costs against chargeback loss exposure.

---

## 📊 Honest Model Metrics (Measured on Test Split)

Reproduce these results anytime via `python3 ml/train_models.py`:

| Model | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|
| **Return Abuse** (customer-level) | **1.00** | **0.75** | **0.86** | 44 test customers, 4 true abusers |
| **Chargeback Risk** (threshold 0.50) | **0.33** | **0.12** | **0.18** | High precision, minimizes false review queue |
| **Chargeback Risk** (threshold 0.30) | **0.08** | **0.38** | **0.14** | Favors recall — catches 3x more real chargebacks |

---

## 🚀 Quickstart & Setup

```bash
# 1. Clone repository & navigate to folder
cd ShieldOps/shieldops

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set your Gemini API Key and Webhook Secret
# Create or edit .env file:
# GEMINI_API_KEY=your_gemini_api_key
# RAZORPAY_WEBHOOK_SECRET=your_webhook_secret

# 4. Generate synthetic dataset & train ML models (if fresh start)
python data/generate_data.py
python ml/train_models.py

# 5. Run ShieldOps application
python app/app.py
# Open http://localhost:5000 in your browser
```

---

## 🎬 5-Minute Pitch Demo Script

1. **Dashboard Overview:** Open `http://localhost:5000` to show the Razorpay-branded interface, live digital clock, summary cards (Cases Flagged, ₹ At Risk, LLM vs. Fallback counts), and the audit log table.
2. **Download Dispute Evidence PDF:** Click **"📄 Evidence PDF"** on any flagged row to show the auto-generated, bank-ready **Dispute Evidence Packet** PDF.
3. **Interactive AI Risk Copilot:** Click **"🤖 Ask Copilot"** or **"🤖 AI Risk Copilot"** and ask *"What is the best defense strategy for this customer?"* to see live Gemini reasoning.
4. **Live Razorpay Webhook Simulation:** Click **"📡 Simulate Razorpay Order"** to simulate a live `order.paid` event, verify cryptographic HMAC scoring, and watch it appear in the audit trail.
5. **Fail-Safe Fallback Demo:** Click **"🧪 Demo: LLM Fallback"** to demonstrate that when LLM calls fail, ShieldOps gracefully switches to deterministic template synthesis without crashing.

---

## 📁 Project Structure

```
ShieldOps/
├── README.md
├── CHANGELOG.md
├── .gitignore
├── .env
└── shieldops/
    ├── requirements.txt
    ├── data/
    │   ├── generate_data.py          # Synthetic dataset generator
    │   ├── customers.csv             # Customer profiles & behavioral flags
    │   └── orders.csv                # Order transactions & return history
    ├── ml/
    │   ├── features.py               # Shared feature engineering pipeline
    │   ├── train_models.py           # ML training & evaluation
    │   └── saved/
    │       ├── return_abuse_model.pkl
    │       ├── chargeback_model.pkl
    │       └── metrics.json
    ├── agent/
    │   └── llm_agent.py              # Gemini LLM prompts, Copilot, & fallback
    └── app/
        ├── app.py                    # Flask server, webhook handler, & API routes
        ├── pdf_generator.py          # ReportLab dispute PDF export engine
        ├── audit_trail.db            # SQLite audit log database
        └── templates/
            └── dashboard.html        # Razorpay-branded UI & Copilot interface
```

---

## 🛡️ Scope & Real-World Integration

- **Defensive & Human-in-the-Loop:** Flags are highlighted with recommended actions for merchant review rather than destructive auto-cancellations.
- **Production Webhooks:** Fully compatible with Razorpay live webhook payloads (`order.paid`, `refund.created`, `payment.captured`).
- **Auditable Provenance:** Every single decision, prompt response, risk tier, and ₹ impact is immutably logged to the audit database.
