import json
import time

import requests

BASE = "http://127.0.0.1:8000"

BENIGN = {"merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI",
          "amount_inr": 1500, "hour_of_day": 13, "txn_count_last_24h": 0,
          "avg_amount_30d_inr": 1600, "distance_from_home_km": 3.0,
          "is_new_device": 0, "account_age_days": 900}
RISKY = {"merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card",
         "amount_inr": 45000, "hour_of_day": 2, "txn_count_last_24h": 9,
         "avg_amount_30d_inr": 3000, "distance_from_home_km": 60.0,
         "is_new_device": 1, "account_age_days": 40}
BIG = dict(BENIGN, merchant_category="Utilities", payment_mode="NetBanking",
           amount_inr=150000, avg_amount_30d_inr=250000, distance_from_home_km=0.0,
           is_new_device=1, account_age_days=2200)

no_hour = {k: v for k, v in BENIGN.items() if k != "hour_of_day"}

# (id, category, method, path, payload, expected status, check on the response body)
CHECKS = [
    ("Q1", "happy", "GET", "/health", None, 200,
     lambda b: b == {"status": "ok"}),
    ("Q2", "happy", "POST", "/score", dict(BENIGN, txn_id="T800001"), 200,
     lambda b: b["flagged"] is False and b["decision"] == "approve" and b["human_review"] is False),
    ("Q3", "escalation", "POST", "/score", dict(RISKY, txn_id="T800002"), 200,
     lambda b: b["flagged"] is True and b["decision"] == "hold_for_review" and b["human_review"] is True),
    ("Q4", "escalation", "POST", "/score", dict(BIG, txn_id="T800003"), 200,
     lambda b: b["flagged"] is False and b["decision"] == "hold_for_review" and b["human_review"] is True),
    ("Q5", "edge", "POST", "/score", dict(BENIGN, amount_inr=-5, txn_id="T800004"), 400,
     lambda b: "error" in b and b.get("field") == "amount_inr"),
    ("Q6", "edge", "POST", "/score", dict(no_hour, txn_id="T800005"), 422,
     lambda b: True),
    ("Q7", "escalation", "POST", "/case",
     dict(RISKY, txn_id="T900002",
          question="more than 6 UPI payments in 24 hours velocity alert"), 200,
     lambda b: b["flagged"] is True and b["attempts"] == 2 and b["safe_default"] is False
     and b["case_note"]["policy_id"] == "P2" and b["human_review"] is True),
    ("Q8", "escalation", "POST", "/case",
     dict(RISKY, txn_id="T900005",
          question="electronics merchant payment on an account younger than 30 days"), 200,
     lambda b: b["attempts"] == 3 and b["safe_default"] is True
     and b["decision"] == "hold_for_review" and b["human_review"] is True),
    ("Q9", "edge", "POST", "/case",
     dict(RISKY, txn_id="T900099",
          question="card payment above INR 50000 from a new device"), 503,
     lambda b: "error" in b),
    ("Q10", "happy", "POST", "/case",
     dict(BENIGN, txn_id="T800001",
          question="card payment above INR 50000 from a new device"), 200,
     lambda b: b["case_note"] is None and b["attempts"] == 0
     and b["decision"] == "approve" and b["human_review"] is False),
    ("Q11", "edge", "POST", "/case",
     dict(RISKY, txn_id="T800006", question="what is the cafeteria menu today"), 200,
     lambda b: b["case_note"] is None and b["attempts"] == 0
     and b["decision"] == "hold_for_review" and b["human_review"] is True),
]


def wait_for_service(limit=120):
    start = time.time()
    while time.time() - start < limit:
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


if __name__ == "__main__":
    if not wait_for_service():
        raise SystemExit("service did not come up within 120 seconds")

    passed = {}
    totals = {}
    print(f"{'id':<5}{'category':<12}{'expected':<10}{'observed':<10}result")
    for qid, cat, method, path, payload, expected, check in CHECKS:
        resp = requests.request(method, BASE + path, json=payload, timeout=30)
        try:
            body = resp.json()
            ok = resp.status_code == expected and bool(check(body))
        except (ValueError, KeyError, TypeError):
            ok = False
        totals[cat] = totals.get(cat, 0) + 1
        passed[cat] = passed.get(cat, 0) + int(ok)
        print(f"{qid:<5}{cat:<12}{expected:<10}{resp.status_code:<10}{'PASS' if ok else 'FAIL'}")

    print()
    for cat in totals:
        print(f"{cat}: {passed[cat]}/{totals[cat]} passed")
    print(f"total: {sum(passed.values())}/{sum(totals.values())} passed")
    print("\n/metrics:")
    print(json.dumps(requests.get(f"{BASE}/metrics", timeout=10).json(), indent=2))