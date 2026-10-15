"""mock_llm.py - offline, deterministic stand-in for a hosted language model.
Needs no API key, no account and no network. Use it unmodified.

    from mock_llm import generate, reset
    text = generate(prompt)      # returns a string (a JSON document, or a defective one)
    reset()                      # forget per-transaction call counts

The prompt must contain these four labelled lines (one per line):
    TXN_ID: <id>
    FRAUD_PROBABILITY: <number between 0 and 1>
    AMOUNT_INR: <number>
    POLICY_ID: <P1 ... P6>
"""
import json
import re

TEAM_BY_POLICY = {"P1": "card_risk", "P2": "upi_risk", "P3": "customer_care",
                  "P4": "merchant_ops", "P5": "merchant_ops", "P6": "customer_care"}

# Defects injected on the first calls for a given TXN_ID (one entry per call).
# After the list is used up, every later call for that TXN_ID is well-formed.
DEFECTS = {
    "T900002": ["bad_json"],
    "T900003": ["missing_field"],
    "T900004": ["bad_value"],
    "T900005": ["bad_json", "bad_json", "bad_json"],
    "T900006": ["approve"],
    "T900007": ["wrong_policy"],
    "T900009": ["bad_json", "missing_field"],
    "T900010": ["bad_team"],
}
_calls = {}


def reset():
    _calls.clear()


def _field(prompt, label):
    m = re.search(rf"^{label}:\s*(\S+)", prompt, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"prompt is missing the line '{label}: ...'")
    return m.group(1)


def generate(prompt, temperature=0.0):
    txn_id = _field(prompt, "TXN_ID")
    if txn_id == "T900099":
        raise ConnectionError("simulated outage of the language-model service")
    prob = float(_field(prompt, "FRAUD_PROBABILITY"))
    amount = float(_field(prompt, "AMOUNT_INR"))
    policy_id = _field(prompt, "POLICY_ID")
    n = _calls.get(txn_id, 0)
    _calls[txn_id] = n + 1
    plan = DEFECTS.get(txn_id, [])
    defect = plan[n] if n < len(plan) else None

    note = {
        "summary": f"Transaction {txn_id} for INR {amount:,.0f} has fraud score {prob:.2f}; policy {policy_id} applies.",
        "priority": "high" if prob >= 0.8 else "medium",
        "recommended_team": TEAM_BY_POLICY.get(policy_id, "customer_care"),
        "recommended_action": "block_and_notify" if policy_id == "P4" else "hold_for_review",
        "policy_id": policy_id,
    }
    if defect == "bad_json":
        return "Here is the case note: " + json.dumps(note)[:-1]      # prose + unterminated JSON
    if defect == "missing_field":
        del note["recommended_team"]
    elif defect == "bad_value":
        note["priority"] = "very urgent"
    elif defect == "bad_team":
        note["recommended_team"] = "money team"
    elif defect == "approve":
        note["recommended_action"] = "approve"
    elif defect == "wrong_policy":
        note["policy_id"] = "P6"
    return json.dumps(note)