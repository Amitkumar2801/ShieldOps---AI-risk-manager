# ShieldOps — Post-Purchase Risk & Dispute Defense Agent

Built for the Razorpay AI Buildathon — **Track 2: AI Risk Manager**

## The problem

Merchants lose money in two connected ways *after* a sale is made:
1. **Serial return abuse** — a small set of customers exploit return policies (wardrobing, multi-accounting) and quietly drain margin.
2. **Chargebacks** — disputed transactions where the merchant loses both the goods and the payment, plus a penalty.

Both are detectable from the same underlying signal: post-purchase behavior. ShieldOps scores both risks from one pipeline and — critically — **explains every flag in plain English** and **drafts the dispute evidence automatically** for high chargeback-risk orders, so a human can act in seconds instead of digging through order history.

## Architecture

```
Synthetic dataset (customers.csv, orders.csv)
        │
        ▼
Feature engineering (ml/features.py)
        │
        ├──► Return Abuse Model (RandomForest, customer-level)
        │
        └──► Chargeback Risk Model (RandomForest, order-level)
                        │
                        ▼
            LLM Agent Layer (Gemini, agent/llm_agent.py)
            - explains WHY a customer was flagged
            - drafts a chargeback evidence packet
            - falls back to a deterministic template if the LLM call fails
                        │
                        ▼
        SQLite Audit Trail + Flask Dashboard (app/app.py)
```

Full data flow, box-by-box, is in the diagram we designed during planning — same shape, now implemented file-for-file.

## Why the design decisions we made

- **Classic ML for detection, LLM only for explanation/drafting.** The risk score itself comes from RandomForest trained on engineered features — this is the right tool for a labeled tabular classification problem. Gemini is used only where language generation actually adds value: explaining a score to a human, and drafting evidence text. This is a deliberate "AI judgment" call, not LLM-for-everything.
- **Threshold is a documented business tradeoff, not a hidden default.** `ml/train_models.py` reports the chargeback model at both threshold 0.50 and 0.30 — recall goes from 12% to 38% but precision drops from 33% to 8%. The live dashboard uses 0.50 to keep the review queue manageable; the 0.30 numbers are kept in `ml/saved/metrics.json` for full transparency.
- **Failure is handled gracefully, by design, not by luck.** Every Gemini call in `agent/llm_agent.py` is wrapped in try/except with an 8-second timeout. If the API key is missing, the network fails, or the response is malformed, ShieldOps **never crashes and never blocks a merchant's workflow** — it falls back to a deterministic template built from the same feature values, and the audit trail logs which path was used (`llm_generated` vs `template_fallback`). Hit the **"Demo: simulate LLM failure"** button on the dashboard to see this live.
- **Every decision is auditable.** Nothing is scored silently — every flagged case, its score, its explanation, its source (LLM or fallback), the recommended action, and the estimated ₹ cost impact are logged to `app/audit_trail.db` and shown on the dashboard.

## Honest metrics (measured, not claimed)

Run `python3 ml/train_models.py` to reproduce these on a fresh train/test split:

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Return Abuse (customer-level) | 1.00 | 0.75 | 0.86 | Only 44 test customers, 4 positives — small sample, honestly reported |
| Chargeback (threshold 0.50) | 0.33 | 0.12 | 0.18 | Default threshold, high precision but misses most chargebacks |
| Chargeback (threshold 0.30) | 0.08 | 0.38 | 0.14 | Recall-favoring tradeoff — more false positives, catches 3x more real chargebacks |

**False-positive cost is quantified, not hand-waved**: at threshold 0.50, 2 false positives cost ~₹100 in wasted manual review; missing 7 real chargebacks cost ₹3,500+ in the test batch alone. Full numbers are in `ml/saved/metrics.json` after training.

This dataset is small and synthetic (150 customers, 612 orders) by design — enough to prove the pipeline end-to-end within the buildathon timeline. With real merchant data (10,000+ transactions), we'd expect materially better recall since the model would see far more positive examples.

## Setup & run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional but recommended) Add your Gemini API key
cp .env.example .env
# edit .env and paste your key, then:
export GEMINI_API_KEY=your_key_here
# Without a key, ShieldOps still works fully — it just uses the template fallback for every explanation.

# 3. Generate the synthetic dataset
python3 data/generate_data.py

# 4. Train both models (prints honest metrics)
python3 ml/train_models.py

# 5. Run the app
python3 app/app.py
# open http://localhost:5000
```

## Demo script (for the 5-minute pitch video)

1. Show the dashboard — flagged return-abuse customers and chargeback-risk orders, with scores, explanations, and ₹ cost impact.
2. Open one flagged customer's explanation — show it's LLM-generated and specific (not generic).
3. Click **"Demo: simulate LLM failure"** — show the batch still completes, every explanation still populates, but the `source` column switches to `template_fallback`. This is the failure-recovery story.
4. Walk through `ml/saved/metrics.json` — show precision/recall/FP-cost honestly, including the threshold tradeoff.
5. Close on the architecture diagram and the one-pipeline design decision.

## What this does NOT do (honest scope limits)

- This is strictly defensive — it never blocks a legitimate customer automatically; every flag is routed to a human for review.
- It does not integrate with a live payment gateway; it batch-scores a synthetic dataset. A production version would consume Razorpay's webhook events in real time.
- The dataset is synthetic; real-world class imbalance and feature richness would change the numbers above.

## Project structure

```
shieldops/
├── data/
│   ├── generate_data.py      # synthetic dataset generator
│   ├── customers.csv
│   └── orders.csv
├── ml/
│   ├── features.py           # shared feature engineering (train + serve)
│   ├── train_models.py       # trains + evaluates both models
│   └── saved/                # trained models + metrics.json
├── agent/
│   └── llm_agent.py          # Gemini calls + graceful fallback
├── app/
│   ├── app.py                # Flask backend + batch scoring + audit trail
│   └── templates/dashboard.html
├── requirements.txt
└── README.md
```
