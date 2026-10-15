"""generate_data.py - synthetic, fictional Razorpay Merchant Risk Desk data.
Run once:  python generate_data.py
Writes transactions_raw.csv, fraud_policies.json, retrieval_queries.json,
case_scenarios.json into the current folder. Standard library only."""
import csv
import json
import math
import random

rng = random.Random(2605)
N_BASE = 6000

CATEGORIES = ["Grocery", "Electronics", "Travel", "Food Delivery", "Fashion", "Utilities"]
CAT_WEIGHTS = [0.24, 0.14, 0.12, 0.22, 0.16, 0.12]
CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata"]
MODES = ["UPI", "Card", "NetBanking", "Wallet"]
MODE_WEIGHTS = [0.55, 0.25, 0.12, 0.08]
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 7, 4, 2]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


rows = []
for i in range(1, N_BASE + 1):
    category = rng.choices(CATEGORIES, weights=CAT_WEIGHTS)[0]
    mode = rng.choices(MODES, weights=MODE_WEIGHTS)[0]
    city = rng.choice(CITIES)
    hour = rng.choices(range(24), weights=HOUR_WEIGHTS)[0]
    avg_amount = round(rng.lognormvariate(8.3, 0.6), 2)            # typical spend, INR
    amount = round(avg_amount * rng.lognormvariate(0.0, 0.75), 2)
    velocity = min(12, int(rng.expovariate(1 / 1.4)))
    distance = round(rng.expovariate(1 / 18.0), 1)
    new_device = 1 if rng.random() < 0.12 else 0
    age_days = rng.randint(5, 2200)
    ratio = amount / avg_amount
    z = (-5.5 + 2.4 * math.log(ratio) + 0.75 * velocity + 2.0 * new_device
         + (2.0 if hour <= 5 else 0.0) + 0.03 * distance
         + (1.0 if category == "Electronics" else 0.0) - 0.0008 * age_days)
    fraud = 1 if rng.random() < sigmoid(z) else 0
    chargeback = (1 if rng.random() < 0.85 else 0) if fraud else (1 if rng.random() < 0.01 else 0)
    rows.append({
        "txn_id": f"T{i:06d}", "customer_id": f"C{rng.randint(1, 1800):04d}",
        "merchant_category": category, "city": city, "payment_mode": mode,
        "amount_inr": amount, "hour_of_day": hour, "txn_count_last_24h": velocity,
        "avg_amount_30d_inr": avg_amount, "distance_from_home_km": distance,
        "is_new_device": new_device, "account_age_days": age_days,
        "chargeback_filed": chargeback, "is_fraud": fraud,
    })


def messy(text):
    r = rng.random()
    if r < 0.10:
        return text.lower()
    if r < 0.18:
        return text.upper()
    if r < 0.26:
        return " " + text + " "
    return text


for r in rows:
    r["merchant_category"] = messy(r["merchant_category"])
    r["city"] = messy(r["city"])

# missing values: distance (fill later), amount (drop later)
for r in rows:
    if rng.random() < 0.03:
        r["distance_from_home_km"] = ""
    if rng.random() < 0.012:
        r["amount_inr"] = ""

# impossible amounts (data-entry errors) on five rows
for idx in rng.sample(range(len(rows)), 5):
    rows[idx]["amount_inr"] = rng.choice([9999999.0, 5000000.0, 0.0, -450.0, 2500000.0])

# exact duplicate rows
for idx in rng.sample(range(len(rows)), 90):
    rows.append(dict(rows[idx]))
rng.shuffle(rows)

with open("transactions_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

POLICIES = [
    {"policy_id": "P1", "title": "High-value card payment review",
     "text": "A card payment above INR 50000 from a device first seen within the last 7 days must be held for review by the card risk team before settlement."},
    {"policy_id": "P2", "title": "UPI velocity limits",
     "text": "More than 6 UPI payments from one customer within 24 hours triggers a velocity alert. The UPI risk team places a temporary hold and contacts the customer to confirm the activity."},
    {"policy_id": "P3", "title": "Night-time and distance anomalies",
     "text": "A payment made between midnight and 5 AM more than 100 km away from the customer home city is flagged. The customer care team calls the customer to verify before the payment is released."},
    {"policy_id": "P4", "title": "Refund and chargeback handling",
     "text": "When a customer reports an unauthorised payment, the merchant operations team opens a chargeback case, blocks further payments on the instrument and sends a notification to the customer."},
    {"policy_id": "P5", "title": "Electronics merchant watchlist",
     "text": "Electronics merchants receive extra monitoring. A payment above INR 30000 on a new account younger than 30 days is held for review by the merchant operations team."},
    {"policy_id": "P6", "title": "Low-risk auto-approval",
     "text": "Payments with a fraud score below the approved threshold on an established account are approved automatically and need no manual review."},
]
with open("fraud_policies.json", "w") as f:
    json.dump(POLICIES, f, indent=2)

RETRIEVAL_QUERIES = [
    {"query": "card payment above INR 50000 from a new device", "expected_policy": "P1"},
    {"query": "how many UPI payments in 24 hours trigger a velocity alert", "expected_policy": "P2"},
    {"query": "payment at 3 AM far away from the home city", "expected_policy": "P3"},
    {"query": "customer reports unauthorised payment and chargeback case", "expected_policy": "P4"},
    {"query": "electronics merchant payment on an account younger than 30 days", "expected_policy": "P5"},
    {"query": "which payments are approved automatically without manual review", "expected_policy": "P6"},
    {"query": "temporary hold on repeated UPI payments within a day", "expected_policy": "P2"},
    {"query": "what happens after a customer reports an unauthorised payment", "expected_policy": "P4"},
    {"query": "what is the cafeteria menu today", "expected_policy": None},
    {"query": "how do I reset my email password", "expected_policy": None},
]
with open("retrieval_queries.json", "w") as f:
    json.dump(RETRIEVAL_QUERIES, f, indent=2)

CASE_SCENARIOS = [
    {"txn_id": "T900001", "fraud_probability": 0.96, "amount_inr": 82000, "question": "card payment above INR 50000 from a new device"},
    {"txn_id": "T900002", "fraud_probability": 0.71, "amount_inr": 6400, "question": "more than 6 UPI payments in 24 hours velocity alert"},
    {"txn_id": "T900003", "fraud_probability": 0.88, "amount_inr": 15500, "question": "payment at 3 AM far away from the home city"},
    {"txn_id": "T900004", "fraud_probability": 0.64, "amount_inr": 3200, "question": "customer reports unauthorised payment and chargeback case"},
    {"txn_id": "T900005", "fraud_probability": 0.91, "amount_inr": 41000, "question": "electronics merchant payment on an account younger than 30 days"},
    {"txn_id": "T900006", "fraud_probability": 0.58, "amount_inr": 9800, "question": "temporary hold on repeated UPI payments within a day"},
    {"txn_id": "T900007", "fraud_probability": 0.77, "amount_inr": 22000, "question": "card payment above INR 50000 from a new device"},
    {"txn_id": "T900008", "fraud_probability": 0.93, "amount_inr": 120000, "question": "what happens after a customer reports an unauthorised payment"},
    {"txn_id": "T900009", "fraud_probability": 0.69, "amount_inr": 7300, "question": "payment at 3 AM far away from the home city"},
    {"txn_id": "T900010", "fraud_probability": 0.84, "amount_inr": 56000, "question": "how many UPI payments in 24 hours trigger a velocity alert"},
]
with open("case_scenarios.json", "w") as f:
    json.dump(CASE_SCENARIOS, f, indent=2)
print("rows written:", len(rows))
