# Fraud Triage Service

## Overview
(baad mein likhenge)

## Reproduce
(baad mein likhenge)

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
## Part C results
## Part D results
## Limitations