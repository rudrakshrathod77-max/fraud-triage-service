# Fraud Triage Service

## Overview
A fraud triage service for a synthetic payments dataset (all data is fictional, amounts in INR).
Part A cleans the raw export and builds three features; Part B trains a tuned random forest and picks an alert threshold on validation data.
Part C retrieves the fraud policy that applies to a flagged payment and asks a (mock) language model for a structured case note, checked by three validation gates.
Part D wraps everything in a FastAPI service with input checks, output guardrails, error handling, one log line per POST request and live metrics.
The model is trained at start-up and kept in memory; nothing is saved to disk.
A flagged payment is never auto-approved, and every held payment goes to a human reviewer.

## Reproduce
Run everything from the repository root (Python 3.11 or newer, shown for macOS/Linux).

```bash
git clone https://github.com/rudrakshrathod77-max/fraud-triage-service.git
cd fraud-triage-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_data.py      # prints: rows written: 6090
python prepare.py            # Part A audit and checkpoints
python train.py              # Part B tables
python retrieval.py          # Part C retrieval table
python casenote.py           # Part C scenario table
```

Part D needs two terminals. Restart the service before every smoke run, because the mock model remembers call counts per transaction id, and delete service.log first so it holds only that run's ten POST lines.

```bash
# terminal 1
rm -f service.log
uvicorn app:app --port 8000

# terminal 2 (after "Application startup complete")
python smoke_test.py
cat service.log
```

Optional extra check (restart the service first, and delete service.log afterwards):

```bash
curl -s -X POST http://127.0.0.1:8000/case -H "Content-Type: application/json" -d '{"txn_id": "T900006", "merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 4500, "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 3000, "distance_from_home_km": 20.0, "is_new_device": 1, "account_age_days": 300, "question": "temporary hold on repeated UPI payments within a day"}'
```

## Part A results

| Check | Value |
|---|---|
| Rows, columns as read | 6,090 / 14 |
| Exact duplicate rows | 90 |
| Rows after de-duplication | 6,000 |
| Missing amount_inr (after dedup) | 76 |
| Missing distance_from_home_km (after dedup) | 193 |
| amount_inr below 1 or above 1,000,000 | 5 |
| Distinct raw spellings, merchant_category | 24 |
| Distinct raw spellings, city | 24 |
| Rows after clean_transactions | 5,919 |
| Remaining missing distance_from_home_km | 191 |
| is_night sum / high_value sum | 365 / 17 |
| Max amount_vs_avg_ratio | 13.11 |

Missing distance_from_home_km values are not filled in Part A because the fill statistic (the median) must be learned from the training rows only, otherwise information from validation and test rows leaks into training.

**Leakage audit (cleaned table)**

| chargeback_filed | Rows | Fraud rows |
|---|---|---|
| 1 | 373 | 304 |
| 0 | 5,546 | 62 |

chargeback_filed is filed only after a payment has settled, so it is not known when the model must decide, and it almost encodes the label (304 of 373 rows are fraud). txn_id and customer_id are identifiers, not behaviour: they carry no real signal and would only let the model memorise rows instead of learning patterns. Excluded columns: chargeback_filed, txn_id, customer_id, is_fraud.

Model-input columns before one-hot encoding (13): merchant_category, city, payment_mode, amount_inr, hour_of_day, txn_count_last_24h, avg_amount_30d_inr, distance_from_home_km, is_new_device, account_age_days, amount_vs_avg_ratio, is_night, high_value.

**Why accuracy misleads.** The cleaned table has 366 fraud rows out of 5,919 (6.18%), and a rule that always predicts "not fraud" gets 93.82% accuracy with recall 0. Accuracy rewards the majority class, so it hides that this rule catches no fraud at all; recall and precision show what matters.

## Part B results

Input columns (26, pinned order):
['amount_inr', 'hour_of_day', 'txn_count_last_24h', 'avg_amount_30d_inr', 'distance_from_home_km', 'is_new_device', 'account_age_days', 'amount_vs_avg_ratio', 'is_night', 'high_value', 'merchant_category_Electronics', 'merchant_category_Fashion', 'merchant_category_Food Delivery', 'merchant_category_Grocery', 'merchant_category_Travel', 'merchant_category_Utilities', 'city_Bengaluru', 'city_Chennai', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'city_Pune', 'payment_mode_Card', 'payment_mode_NetBanking', 'payment_mode_UPI', 'payment_mode_Wallet']

**Split (stratified 60/20/20, random_state=42)**

| Split | Rows | Fraud rows |
|---|---|---|
| Train | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

Training median of distance_from_home_km: 12.6 (used to fill missing values in all three splits).

**Three ways to treat imbalance (logistic regression, validation, threshold 0.5)**

| Variant | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| Plain | 0.625 | 0.342 | 0.442 | 1096 | 15 | 48 | 25 |
| class_weight=balanced | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| Random over-sampling | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

The over-sampled training table has 6,662 rows (3,331 non-fraud and 3,331 fraud).

A risk desk that must not miss fraud would prefer a balanced or over-sampled model, because recall rises from 0.342 to 0.781 (57 frauds caught instead of 25). The cost is precision, which falls from 0.625 to about 0.25, so roughly 165 to 169 legitimate payments are held instead of 15, and analysts get far more work.

**Cross-validation (5 folds, training split only, F1)**

| Model | Mean F1 | Std |
|---|---|---|
| Balanced logistic regression | 0.402 | 0.015 |
| Random forest (200 trees, balanced), untuned | 0.453 | 0.052 |
| Tuned random forest | 0.464 | 0.042 |

GridSearchCV tried 12 combinations. Best parameters: max_depth=None, min_samples_leaf=5 (n_estimators=200 fixed). Best mean CV F1: 0.464.

The tuned forest is the final model because it has the best CV F1 (0.464 against 0.402 for logistic regression), and it can use non-linear interactions such as a high ratio combined with high velocity. Its fold-to-fold spread (std 0.042) is larger than that of logistic regression (0.015), so it is less stable, and the gain over the untuned forest (0.453) is small. For interpretability it gives feature importances, which are easy to read but less transparent than the signed coefficients of logistic regression. Compared with logistic regression I give up stability, simplicity and an explanation of direction for each feature, in exchange for higher accuracy.

**Validation thresholds (tuned forest)**

| Threshold | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| 0.1 | 0.111 | 0.973 | 0.200 | 637 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| 0.4 | 0.271 | 0.712 | 0.392 | 192 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.737 | 0.192 | 0.304 | 19 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

t_star = 0.4 (highest threshold with validation recall at least 0.65).

**Test result at t_star = 0.4 (evaluated once)**

| Precision | Recall | F1 | TN | FP | FN | TP | Flagged |
|---|---|---|---|---|---|---|---|
| 0.282 | 0.781 | 0.415 | 966 | 145 | 16 | 57 | 202 |

**Risk-committee note.** The threshold t_star is the highest value in the 0.1 to 0.9 grid whose validation recall is at least 0.65, which gave 0.4 (validation recall 0.712, while 0.5 gives only 0.589). On the 1,184 test payments the final model flagged 202 payments at t_star. It caught 57 of 73 frauds (recall 0.781, precision 0.282), so 145 legitimate payments were held for 57 frauds caught, about 2.54 legitimate holds per fraud caught. 16 frauds were missed. A stricter threshold of 0.6 would flag only 73 validation payments instead of 192 and raise precision from 0.271 to 0.507, but recall would fall from 0.712 to 0.507, about 15 more frauds missed out of 73.

**Leakage experiment (validation F1 at 0.5)**

| Model | Validation F1 |
|---|---|
| Clean tuned forest (26 columns) | 0.441 |
| Leaky forest (+ chargeback_filed as 27th column) | 0.797 |

The leaky score is not something to be proud of because chargeback_filed is only known after a payment has settled and been disputed, so it is not available when the desk must decide. It nearly copies the label, so the model would look excellent in testing and then lose that signal completely in live use.

**Top 5 features (tuned forest)**

| Feature | Importance |
|---|---|
| amount_vs_avg_ratio | 0.2702 |
| txn_count_last_24h | 0.1778 |
| amount_inr | 0.1337 |
| account_age_days | 0.0835 |
| distance_from_home_km | 0.0708 |

The top two features show that fraudsters spend far more than the customer normally does and fire many payments in a short window before the account is blocked.

## Part C results

**Retrieval (TF-IDF, default settings, cosine similarity)**

| Query | Expected | Retrieved | Score |
|---|---|---|---|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | None | P3 | 0.4251 |
| how do I reset my email password | None | P1 | 0.0000 |

In-scope top-1 hits: 8 of 8.

**MIN_SCORE.** The highest out-of-scope score is 0.4251 and the lowest in-scope score is 0.4862, so MIN_SCORE must lie above 0.4251 and at most 0.4862; I chose 0.45. Queries scoring below it get status no_policy_match and are never sent to the language model.

**Scenario run (mock_llm reset, 10 scenarios in file order)**

| Scenario | Retrieved policy | Attempts | Failures | Status | Safe default | Final action | Override |
|---|---|---|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

Totals: 19 model calls; 3 scenarios valid on the first call (S1, S6, S8); 6 recovered by retry (S2, S3, S4, S7, S9, S10); 1 safe default (S5); 1 guardrail override (S6).

A transaction with no matching policy returns status no_policy_match with attempts 0 and case_note None without calling the model, and a ConnectionError for T900099 propagates out of run_case.

A hosted model samples its words, so the same prompt can come back as broken JSON, a missing key or a wrong value on two calls in a row, and a provider update or heavy load can change its behaviour without warning. Because a fluent answer can still be malformed or wrong (S6 first said approve for a 0.58 fraud score), nothing the model writes can be trusted blindly. That is why every note goes through the three validation gates, a bounded retry, a safe default and the approve-override business rule.

## Part D results

**Smoke test (fresh service, 11 of 11 PASS: happy 3/3, escalation 4/4, edge 4/4)**

| Id | Category | Expected | Observed | Result |
|---|---|---|---|---|
| Q1 | happy | 200 | 200 | PASS |
| Q2 | happy | 200 | 200 | PASS |
| Q3 | escalation | 200 | 200 | PASS |
| Q4 | escalation | 200 | 200 | PASS |
| Q5 | edge | 400 | 400 | PASS |
| Q6 | edge | 422 | 422 | PASS |
| Q7 | escalation | 200 | 200 | PASS |
| Q8 | escalation | 200 | 200 | PASS |
| Q9 | edge | 503 | 503 | PASS |
| Q10 | happy | 200 | 200 | PASS |
| Q11 | edge | 200 | 200 | PASS |

**GET /metrics after that run**

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 7.0,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

**Three lines from service.log (one per level)**

```
2026-10-15 09:11:10,150 INFO endpoint=/score status=200 latency_ms=6.8 txn_id=T800001 fraud_probability=0.0430
2026-10-15 09:11:10,164 WARNING endpoint=/score status=200 latency_ms=6.1 txn_id=T800003 fraud_probability=0.1813 guardrail_override=true
2026-10-15 09:11:10,205 ERROR endpoint=/case status=503 latency_ms=20.1 txn_id=T900099 fraud_probability=0.9596
```

service.log holds exactly the ten POST lines of one smoke run: INFO for Q2, Q3, Q7, Q10, Q11; WARNING for Q4, Q5, Q6, Q8; ERROR for Q9.

**Extra check (fresh service, txn T900006).** The model's first reply says approve, the fraud probability is 0.4391 (flagged, below 0.5), and the service answers 200 with decision hold_for_review, case_note.recommended_action hold_for_review, guardrail_override true and human_review true.

## Limitations

- The data is synthetic and the fraud rule that generated it is hand-built, so the scores say little about real payment fraud.
- The threshold of 0.4 was tuned on only 73 validation fraud cases, so it is a noisy estimate and could shift with a different split.
- The retriever is keyword-based TF-IDF and cannot match meaning, so a paraphrased question with different words can fall below MIN_SCORE or hit the wrong policy.
- The language model is a scripted mock with fixed defects, so the retry and validation logic was never tested against the variety of a real hosted model.
- The model is retrained at every start-up (about a minute) and held in memory, so the service restarts slowly and the counters and logs reset with it.