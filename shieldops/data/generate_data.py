"""
ShieldOps — Synthetic Data Generator
=====================================
Generates a realistic e-commerce dataset with TWO ground-truth risk patterns
planted into it, so we can later train models and honestly measure
precision/recall against known labels:

  1. Return abuse    -> customer-level pattern (serial returners, wardrobing,
                         multi-accounting via shared address)
  2. Chargeback risk -> order-level pattern (undelivered/disputed high-value
                         orders, new-device velocity spikes)

Run:  python3 generate_data.py
Output: customers.csv, orders.csv  (in the same folder)
"""

import random
import csv
import datetime as dt

random.seed(42)

N_CUSTOMERS = 150
N_ADDRESSES = 40          # fewer addresses than customers -> some sharing happens
START_DATE = dt.date(2026, 6, 1)
DAYS_WINDOW = 90

FIRST_NAMES = ["Aarav","Vivaan","Aditya","Priya","Ananya","Diya","Rohan","Kabir",
               "Ishaan","Sara","Meera","Arjun","Kavya","Nikhil","Riya","Yash",
               "Tanvi","Aryan","Zoya","Dev"]
LAST_NAMES = ["Sharma","Verma","Gupta","Iyer","Reddy","Khan","Patel","Singh",
              "Nair","Rao","Mishra","Joshi","Kapoor","Das","Chatterjee"]

CATEGORIES = [
    ("Electronics", 3000, 45000),
    ("Fashion", 500, 6000),
    ("Home & Kitchen", 400, 8000),
    ("Beauty", 200, 3000),
    ("Grocery", 150, 2500),
]

RETURN_REASONS = ["Size issue", "Not as described", "Changed mind",
                   "Damaged item", "Wrong item sent", "Better price found"]

CHARGEBACK_REASONS = ["Item not received", "Unauthorized transaction",
                       "Not as described", "Duplicate charge"]

# ---------- Step 1: Addresses (some shared to simulate multi-accounting) ----
addresses = [f"addr_{i:03d}" for i in range(N_ADDRESSES)]

# ---------- Step 2: Customer archetypes -------------------------------------
# normal          : low return rate, near-zero chargeback risk
# occasional_issue: moderate return rate, small chargeback risk
# abuser          : high return rate, wardrobing, higher chargeback risk,
#                   some paired on the same address (multi-accounting)
archetype_weights = [("normal", 0.70), ("occasional_issue", 0.20), ("abuser", 0.10)]


def pick_archetype():
    r = random.random()
    cum = 0
    for name, w in archetype_weights:
        cum += w
        if r <= cum:
            return name
    return "normal"


customers = []
abuser_pool_for_pairing = []

for i in range(N_CUSTOMERS):
    cust_id = f"CUST{i:04d}"
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    archetype = pick_archetype()
    signup_date = START_DATE - dt.timedelta(days=random.randint(0, 400))

    if archetype == "abuser":
        abuser_pool_for_pairing.append(cust_id)

    customers.append({
        "customer_id": cust_id,
        "name": name,
        "archetype": archetype,          # kept for reference / eval only
        "signup_date": signup_date.isoformat(),
        "address_id": random.choice(addresses),
    })

# Force ~5 pairs of abusers to intentionally share the SAME address
# (classic multi-accounting signal)
random.shuffle(abuser_pool_for_pairing)
pair_addr_pool = random.sample(addresses, min(6, len(abuser_pool_for_pairing) // 2))
pi = 0
for j in range(0, len(abuser_pool_for_pairing) - 1, 2):
    if pi >= len(pair_addr_pool):
        break
    shared_addr = pair_addr_pool[pi]
    a, b = abuser_pool_for_pairing[j], abuser_pool_for_pairing[j + 1]
    for c in customers:
        if c["customer_id"] in (a, b):
            c["address_id"] = shared_addr
    pi += 1

cust_by_id = {c["customer_id"]: c for c in customers}

# ---------- Step 3: Orders per customer -------------------------------------
ARCHETYPE_PARAMS = {
    "normal": dict(n_orders=(1, 6), return_prob=0.10, cb_prob=0.015,
                   return_days=(2, 10)),
    "occasional_issue": dict(n_orders=(2, 8), return_prob=0.28, cb_prob=0.05,
                              return_days=(3, 15)),
    "abuser": dict(n_orders=(3, 10), return_prob=0.65, cb_prob=0.15,
                    return_days=(18, 29)),  # returns filed close to policy limit
}

orders = []
order_counter = 0

for c in customers:
    params = ARCHETYPE_PARAMS[c["archetype"]]
    n_orders = random.randint(*params["n_orders"])

    for _ in range(n_orders):
        order_counter += 1
        order_id = f"ORD{order_counter:05d}"
        order_date = START_DATE + dt.timedelta(days=random.randint(0, DAYS_WINDOW))
        category, lo, hi = random.choice(CATEGORIES)
        order_value = round(random.uniform(lo, hi), 2)

        # abusers disproportionately target high-value items to return
        if c["archetype"] == "abuser" and random.random() < 0.5:
            order_value = round(order_value * random.uniform(1.3, 2.0), 2)

        delivery_confirmed = random.random() > (0.08 if c["archetype"] != "abuser" else 0.20)
        payment_method = random.choice(["UPI", "Card", "NetBanking", "Wallet"])
        new_device = random.random() < (0.10 if c["archetype"] != "abuser" else 0.30)

        is_returned = random.random() < params["return_prob"]
        return_date, return_reason, days_to_return = "", "", ""
        if is_returned:
            d = random.randint(*params["return_days"])
            days_to_return = d
            return_date = (order_date + dt.timedelta(days=d)).isoformat()
            return_reason = random.choice(RETURN_REASONS)

        # Chargeback more likely when delivery NOT confirmed + high value + new device
        cb_base = params["cb_prob"]
        if not delivery_confirmed:
            cb_base += 0.20
        if order_value > 15000:
            cb_base += 0.05
        if new_device:
            cb_base += 0.05
        is_chargeback = (not is_returned) and (random.random() < min(cb_base, 0.9))
        chargeback_reason = random.choice(CHARGEBACK_REASONS) if is_chargeback else ""

        orders.append({
            "order_id": order_id,
            "customer_id": c["customer_id"],
            "order_date": order_date.isoformat(),
            "order_value": order_value,
            "category": category,
            "payment_method": payment_method,
            "delivery_confirmed": int(delivery_confirmed),
            "new_device": int(new_device),
            "is_returned": int(is_returned),
            "return_date": return_date,
            "return_reason": return_reason,
            "days_to_return": days_to_return,
            "is_chargeback": int(is_chargeback),
            "chargeback_reason": chargeback_reason,
        })

# ---------- Step 4: Ground-truth customer-level abuse label -----------------
# A customer is a "true return abuser" if their archetype is abuser AND they
# actually exhibit a high return rate in the generated data (keeps labels
# grounded in the data itself, not just the hidden archetype).
from collections import defaultdict

cust_orders = defaultdict(list)
for o in orders:
    cust_orders[o["customer_id"]].append(o)

for c in customers:
    co = cust_orders.get(c["customer_id"], [])
    n = len(co)
    n_returns = sum(o["is_returned"] for o in co)
    return_rate = n_returns / n if n else 0
    c["true_return_abuser"] = int(c["archetype"] == "abuser" and return_rate >= 0.4)

# ---------- Step 5: Write CSVs ----------------------------------------------
with open("/home/claude/shieldops/data/customers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["customer_id", "name", "archetype",
                                       "signup_date", "address_id", "true_return_abuser"])
    w.writeheader()
    w.writerows(customers)

with open("/home/claude/shieldops/data/orders.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(orders[0].keys()))
    w.writeheader()
    w.writerows(orders)

print(f"Generated {len(customers)} customers and {len(orders)} orders.")
print(f"True return abusers (ground truth): {sum(c['true_return_abuser'] for c in customers)}")
print(f"Chargeback orders (ground truth): {sum(o['is_chargeback'] for o in orders)}")
