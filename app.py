import logging
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from casenote import run_case
from prepare import add_features
from train import build_final_model

# ---- logging: one file, written only from the middleware ----
_handler = logging.FileHandler("service.log")
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log = logging.getLogger("triage")
log.setLevel(logging.INFO)
log.addHandler(_handler)
log.propagate = False

# ---- model, built once at start-up ----
MODEL = build_final_model()
FOREST = MODEL["model"]
COLUMNS = MODEL["columns"]
MEDIAN = MODEL["median"]
THRESHOLD = MODEL["threshold"]

CATEGORIES = {"Grocery", "Electronics", "Travel", "Food Delivery", "Fashion", "Utilities"}
UNAVAILABLE = {"error": "case-note service temporarily unavailable, try again later"}

STATS = {"requests_total": 0, "success_total": 0, "error_total": 0,
         "latency_sum": 0.0, "human_review_total": 0}

app = FastAPI()


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


# ---- the one place that logs and counts every POST response ----
def record(path, status, ms, info):
    STATS["requests_total"] += 1
    STATS["success_total" if status < 400 else "error_total"] += 1
    STATS["latency_sum"] += ms
    if status < 400 and info.get("human_review"):
        STATS["human_review_total"] += 1

    if status >= 500:
        level = logging.ERROR
    elif status >= 400 or info.get("safe_default") or info.get("override"):
        level = logging.WARNING
    else:
        level = logging.INFO

    msg = f"endpoint={path} status={status} latency_ms={ms:.1f}"
    if "txn_id" in info:
        msg += f" txn_id={info['txn_id']}"
    if "probability" in info:
        msg += f" fraud_probability={info['probability']:.4f}"
    if info.get("safe_default"):
        msg += " safe_default=true"
    if info.get("override"):
        msg += " guardrail_override=true"
    log.log(level, msg)


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.info = {}
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse(status_code=500, content={"error": "internal server error"})
    if request.method == "POST":
        ms = (time.perf_counter() - t0) * 1000
        record(request.url.path, response.status_code, ms, request.state.info)
    return response


# ---- helpers ----
def reject(field, message):
    return JSONResponse(status_code=400, content={"error": message, "field": field})


def check_input(t):
    if not (1 <= t.amount_inr <= 1_000_000):
        return reject("amount_inr", "amount_inr must be between 1 and 1,000,000")
    if not (0 <= t.hour_of_day <= 23):
        return reject("hour_of_day", "hour_of_day must be between 0 and 23")
    if t.txn_count_last_24h < 0:
        return reject("txn_count_last_24h", "txn_count_last_24h must not be negative")
    if t.avg_amount_30d_inr <= 0:
        return reject("avg_amount_30d_inr", "avg_amount_30d_inr must be greater than 0")
    if t.merchant_category.strip().title() not in CATEGORIES:
        return reject("merchant_category", "unknown merchant_category")
    return None


def fraud_probability(t):
    row = t.model_dump()
    row.pop("question", None)
    df = pd.DataFrame([row])
    df["merchant_category"] = df["merchant_category"].str.strip().str.title()
    df["city"] = df["city"].str.strip().str.title()
    df = add_features(df)
    df = df.drop(columns=["txn_id"])
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=COLUMNS, fill_value=0)
    df["distance_from_home_km"] = pd.to_numeric(
        df["distance_from_home_km"], errors="coerce").fillna(MEDIAN)
    return float(FOREST.predict_proba(df)[0, 1])


def unflagged_decision(t, info):
    """Decision for a payment the model did not flag (includes the large-payment hold)."""
    if t.amount_inr >= 100_000 and t.is_new_device == 1:
        info["override"] = True
        return "hold_for_review", True
    return "approve", False


# ---- endpoints ----
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    n = STATS["requests_total"]

    def rate(x):
        return round(x / n, 4) if n else 0.0

    return {"requests_total": n,
            "success_total": STATS["success_total"],
            "error_total": STATS["error_total"],
            "success_rate": rate(STATS["success_total"]),
            "error_rate": rate(STATS["error_total"]),
            "avg_latency_ms": round(STATS["latency_sum"] / n, 2) if n else 0.0,
            "human_review_total": STATS["human_review_total"],
            "human_review_rate": rate(STATS["human_review_total"])}


@app.post("/score")
def score(t: Txn, request: Request):
    bad = check_input(t)
    if bad:
        return bad
    info = request.state.info
    info["txn_id"] = t.txn_id
    prob = fraud_probability(t)
    info["probability"] = prob
    flagged = prob >= THRESHOLD
    if flagged:
        decision, review = "hold_for_review", True
    else:
        decision, review = unflagged_decision(t, info)
    info["human_review"] = review
    return {"txn_id": t.txn_id, "fraud_probability": round(prob, 4), "flagged": flagged,
            "decision": decision, "human_review": review}


@app.post("/case")
def case(c: CaseIn, request: Request):
    bad = check_input(c)
    if bad:
        return bad
    info = request.state.info
    info["txn_id"] = c.txn_id
    prob = fraud_probability(c)
    info["probability"] = prob
    flagged = prob >= THRESHOLD
    out = {"txn_id": c.txn_id, "fraud_probability": round(prob, 4), "flagged": flagged}

    if not flagged:
        decision, review = unflagged_decision(c, info)
        info["human_review"] = review
        return {**out, "decision": decision, "human_review": review, "case_note": None,
                "attempts": 0, "safe_default": False, "guardrail_override": False}

    try:
        result = run_case(c.txn_id, prob, c.amount_inr, c.question)
    except ConnectionError:
        return JSONResponse(status_code=503, content=UNAVAILABLE)

    if result["status"] == "no_policy_match":
        info["human_review"] = True
        return {**out, "decision": "hold_for_review", "human_review": True,
                "case_note": None, "attempts": 0, "safe_default": False,
                "guardrail_override": False}

    note = result["case_note"]
    override = result["guardrail_override"]
    if note["recommended_action"] == "approve":      # a flagged payment is never auto-approved
        note["recommended_action"] = "hold_for_review"
        override = True
    decision = note["recommended_action"]
    review = decision in ("hold_for_review", "block_and_notify") or result["safe_default"]
    info["human_review"] = review
    info["safe_default"] = result["safe_default"]
    info["override"] = override
    return {**out, "decision": decision, "human_review": review, "case_note": note,
            "attempts": result["attempts"], "safe_default": result["safe_default"],
            "guardrail_override": override}