import json

import mock_llm
from retrieval import MIN_SCORE, retrieve

CASE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string",
                             "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string",
                               "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]},
    },
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"],
}

SAFE_DEFAULT = {
    "summary": "Automated case note unavailable; manual review required.",
    "priority": "high",
    "recommended_team": "customer_care",
    "recommended_action": "hold_for_review",
}


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    props = CASE_NOTE_SCHEMA["properties"]
    return f"""ROLE
You are a fraud-risk assistant at a payments company. You write short case notes for human analysts.

TASK
Write a case note for the held payment below, based on the fraud policy that applies to it.
TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}
Policy text: {policy_text}

CONSTRAINTS
- Use only the policy text given above; do not invent rules.
- Reply with one JSON object and nothing else: no prose, no markdown, no code fences.

OUTPUT FORMAT
The JSON object has exactly these five keys:
- summary: one short sentence (string)
- priority: one of {props['priority']['enum']}
- recommended_team: one of {props['recommended_team']['enum']}
- recommended_action: one of {props['recommended_action']['enum']}
- policy_id: one of {props['policy_id']['enum']}
"""


def validate(text, retrieved_policy_id):
    try:                                                  # Gate 1: valid JSON object
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1
    if not isinstance(note, dict):
        return None, 1
    for key in CASE_NOTE_SCHEMA["required"]:              # Gate 2: keys present, all strings
        if key not in note or not isinstance(note[key], str):
            return None, 2
    for key, rule in CASE_NOTE_SCHEMA["properties"].items():   # Gate 3: allowed values
        if "enum" in rule and note[key] not in rule["enum"]:
            return None, 3
    if note["policy_id"] != retrieved_policy_id:
        return None, 3
    return note, 0


def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    policy_id, policy_text, score = retrieve(question)
    if score < MIN_SCORE:
        return {"status": "no_policy_match", "attempts": 0, "failures": [],
                "case_note": None, "safe_default": False, "guardrail_override": False}

    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)
    failures, note, attempts = [], None, 0
    for _ in range(max_attempts):
        text = mock_llm.generate(prompt)       # ConnectionError is NOT caught here
        attempts += 1
        note, gate = validate(text, policy_id)
        if gate == 0:
            break
        failures.append(gate)
        note = None

    safe_default = note is None
    if safe_default:
        note = dict(SAFE_DEFAULT, policy_id=policy_id)

    override = False
    if note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        note["recommended_action"] = "hold_for_review"
        override = True

    return {"status": "ok", "attempts": attempts, "failures": failures,
            "case_note": note, "safe_default": safe_default, "guardrail_override": override}


if __name__ == "__main__":
    mock_llm.reset()
    with open("case_scenarios.json") as f:
        scenarios = json.load(f)

    print(f"{'scenario':<14}{'policy':<8}{'attempts':<10}{'failures':<12}{'status':<9}"
          f"{'safe default':<14}{'final action':<18}override")
    calls = first_ok = recovered = safe = overrides = 0
    for i, sc in enumerate(scenarios, 1):
        r = run_case(sc["txn_id"], sc["fraud_probability"], sc["amount_inr"], sc["question"])
        policy = r["case_note"]["policy_id"] if r["case_note"] else "-"
        action = r["case_note"]["recommended_action"] if r["case_note"] else "-"
        print(f"S{i} ({sc['txn_id']})".ljust(14) + f"{policy:<8}{r['attempts']:<10}"
              f"{str(r['failures']):<12}{r['status']:<9}"
              f"{'yes' if r['safe_default'] else 'no':<14}{action:<18}"
              f"{'yes' if r['guardrail_override'] else 'no'}")
        calls += r["attempts"]
        first_ok += r["attempts"] == 1 and not r["failures"]
        recovered += bool(r["failures"]) and not r["safe_default"]
        safe += r["safe_default"]
        overrides += r["guardrail_override"]

    print(f"\nTotals: {calls} model calls; {first_ok} valid on first call; "
          f"{recovered} recovered by retry; {safe} safe default; {overrides} guardrail override")