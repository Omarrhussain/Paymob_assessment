# Paymob Data Science Intern Assessment — PRD & Execution Plan

**Author:** Omar Hussain  
**Date:** June 2026  
**Assessment:** Paymob DS Intern Take-Home · Python · Synthetic Data

---

## 1. Overview

This document defines the plan, data findings, and execution roadmap for the Paymob DS Intern Assessment. The assessment has three graded tasks (Digest, Reach, Reach·ROI) and one optional bonus (Copilot·GenAI). The deliverables mirror three real Paymob products.

---

## 2. Data Audit — What We Found

Before any analysis, a full quality scan was run on both files. Below are **all confirmed issues** with the planned fix for each.

### 2.1 `transactions.csv` — 107,501 rows × 11 columns

| # | Issue | Severity | Detail | Fix |
|---|-------|----------|--------|-----|
| 1 | **Case-inconsistent `payment_method`** | High | `wallet`, `Wallet`, `WALLET`, ` wallet` (leading space) all appear. Same for card, instapay, installment, cod. 18 distinct raw values should map to 5. | `.str.strip().str.lower()` + map `cod` / `cash_on_delivery` → `cash_on_delivery`. |
| 2 | **534 duplicate `transaction_id` rows** | High | Exact duplicates (same id, customer, amount, timestamp). Classic export double-counting bug. | `drop_duplicates(subset='transaction_id', keep='first')`. After dedup: **106,967 rows**. |
| 3 | **213 negative `amount_egp` values** | High | Min = –14,204 EGP. Negatives on non-refunded rows are data errors; negatives on refunded rows may be correct representation. | Flag separately: for `status == 'refunded'`, use `abs(amount_egp)`. For `status == 'success'` with negative amount → drop or flag as corrupt. |
| 4 | **2,688 null `city` values** | Medium | ~2.5% missing. Not ignorable for geo analysis. | Keep rows; label nulls as `"Unknown"` for segment/trend work. Do not impute from merchant_id (no reliable mapping). |
| 5 | **286 rows with `failure_reason` populated but `status ≠ failed`** | Medium | Orphaned failure reasons on success/refunded rows. Likely export artifact. | Set `failure_reason = NaN` where `status != 'failed'`. |
| 6 | **Timezone not applied** | Medium | `timestamp_utc` is stored in UTC but Egypt operates at UTC+2 (Africa/Cairo). Peak-time analysis will be off by 2 hours if not converted. | `pd.to_datetime(..., utc=True).dt.tz_convert('Africa/Cairo')` for all time-of-day analysis. |
| 7 | **Extreme `amount_egp` outliers** | Low | Top values: 7.3M, 7.3M, 6.8M EGP — orders of magnitude above the median (485 EGP). Likely B2B or test transactions. | Keep in GMV totals (they are legitimate settled revenue). Exclude from customer spend segmentation percentile cuts (use IQR-capped values for segment thresholds only). |
| 8 | **`timestamp_utc` date range** | Low | Data starts 2024-04-30, not May 2024 as stated in the dictionary. One day overlap. | Note in assumptions. No rows dropped — edge case doesn't affect analysis. |

### 2.2 `campaign_responses.csv` — 1,800 rows × 4 columns

| # | Issue | Severity | Detail | Fix |
|---|-------|----------|--------|-----|
| 1 | **Class imbalance** | High | 423 responded (23.5%) vs 1,377 did not (76.5%). Plain accuracy is misleading. | Use ROC-AUC and F1 (macro or weighted) as primary metrics. Consider `class_weight='balanced'` in model. |
| 2 | **Leakage risk** | Critical | Campaign sent 2025-04-15. Any transaction feature built from data **after** this date would leak the label. | Hard cutoff: all feature engineering uses only transactions with `timestamp_utc < 2025-04-15`. |
| 3 | **All 1,800 targeted customers exist in transactions** | ✅ Clean | No orphan joins. |  |
| 4 | **Single campaign, single date** | Low | No campaign-level variation to exploit. | Treat `campaign_name` and `campaign_date` as constants; drop from model features. |

---

## 3. Revenue / GMV Definition

**Settled GMV = sum of `amount_egp` where `status = 'success'`.**

- `refunded` rows represent money that left the merchant — excluded from GMV.
- `failed` rows never settled — excluded.
- For refund-rate metrics, numerator = refunded transaction count; denominator = success + refunded.
- Negative amounts on success rows (data error) are dropped before GMV calculation.

This definition is stated explicitly in the notebook as required by the rubric.

---

## 4. Task Breakdown & Execution Plan

### Task 1 — DIGEST: Merchant & Customer Insights

**Goal:** Produce the analytical story a Digest dashboard would tell, written for a business stakeholder.

**Deliverable:** 5–8 key findings + supporting charts/tables.

#### Steps

1. **Data cleaning** (as per Section 2). Document every decision inline.

2. **Headline KPIs**
   - Total settled GMV (EGP)
   - Total transaction volume (success + failed + refunded)
   - Success rate, failure rate, refund rate
   - Average ticket (mean `amount_egp` on success rows)

3. **Monthly trends**
   - Group by year-month on Cairo local time
   - Plot GMV and volume over time
   - Flag: Ramadan 2024 (Mar 11 – Apr 9) and Ramadan 2025 (Mar 1 – Mar 30) — look for spending pattern shifts
   - Flag: back-to-school (Aug–Sep), end-of-year (Dec)

4. **Peak selling times**
   - Convert timestamps to Africa/Cairo
   - Heatmap: hour-of-day × day-of-week for transaction volume
   - Separate online vs POS — they likely have different patterns

5. **Payment method mix**
   - After normalizing method names: share by volume and by GMV
   - Failure rate per method — which is least reliable?

6. **Refund rates by MCC category**
   - `refunded / (success + refunded)` per category
   - Rank by risk; call out top-3 categories

7. **Customer spend segments**
   - Per customer: total settled spend (success rows only, post-dedup)
   - Segment thresholds (proposed, to be validated against distribution):
     - **High:** top 20% by total spend
     - **Mid:** 20th–60th percentile
     - **Mass:** bottom 40%
   - Report count, GMV share, and avg ticket per segment

**Key Findings Format:** Each finding = one sentence headline + one supporting number + one business implication.

---

### Task 2 — REACH: Customer Segmentation

**Goal:** RFM segmentation → actionable named segments → target list CSV.

#### Steps

1. **RFM Feature Engineering** (on success transactions only, full date range)
   - **Recency:** days since last successful transaction (reference date = 2025-06-30, last date in data)
   - **Frequency:** count of successful transactions
   - **Monetary:** total settled spend (EGP)

2. **Scoring** — quintile-based (1–5 per dimension, 5 = best)

3. **Segment Definitions**

   | Segment | RFM Logic | Business Label |
   |---------|-----------|----------------|
   | Champions | R≥4, F≥4, M≥4 | Champions |
   | Loyal | F≥4, M≥3 | Loyal Customers |
   | Potential Loyalists | R≥3, F=2-3 | Potential Loyalists |
   | At-Risk | R=2-3, F≥3, M≥3 | At-Risk |
   | Hibernating | R≤2, F≤2 | Hibernating |
   | Lost | R=1, F=1 | Lost |

4. **Segment to Target:** Recommend **At-Risk** — they have proven spend history (high M, high F) but declining recency. Win-back campaigns are high-ROI on this group. Back this with: average historical spend vs Champions, time since last purchase distribution.

5. **Export:** `target_customers_at_risk.csv` with `customer_id`, RFM scores, and segment label.

---

### Task 3 — REACH·ROI: Campaign Response Prediction

**Goal:** Predict `responded` (binary) for targeted customers. No leakage. Proper validation.

#### Feature Engineering (all using transactions before 2025-04-15 only)

| Feature | Description |
|---------|-------------|
| `recency_days` | Days since last successful tx before campaign |
| `frequency_90d` | Tx count in 90 days before campaign |
| `frequency_180d` | Tx count in 180 days before campaign |
| `total_spend` | Lifetime settled spend |
| `avg_ticket` | Mean amount per successful tx |
| `spend_90d` | Total spend in 90 days before campaign |
| `active_months` | Count of distinct months with ≥1 success tx |
| `top_mcc` | Most frequent MCC category (one-hot or label encode) |
| `payment_method_diversity` | Count of distinct methods used |
| `online_ratio` | Share of online vs POS |
| `failure_rate` | Failed / total attempts |
| `city` | One-hot or target-encode |

**No features derived from post-2025-04-15 transactions.**

#### Modelling

- **Train/test split:** Stratified 80/20 on `responded`
- **Models:** Logistic Regression (baseline) → Random Forest or XGBoost (main)
- **Primary metric:** ROC-AUC (imbalance-robust). Secondary: F1-macro, Precision@K
- **Hyperparameter tuning:** 5-fold stratified cross-validation on train set
- **Class imbalance handling:** `class_weight='balanced'` or SMOTE on train fold only

#### Honest Read

Report: AUC, F1, confusion matrix on held-out test set. State clearly what the model can and cannot do. Suggest how Reach can use predicted probabilities to rank customers and prioritize high-response-probability targets in future campaigns.

---

### Task 4 (Bonus) — COPILOT: Explain the Failures (GenAI)

**Goal:** Identify merchants with abnormal failure patterns; build a Copilot feature to explain failures in plain English.

#### Steps

1. **Anomaly detection (analytical)**
   - Per-merchant failure rate vs dataset mean
   - Z-score or IQR flagging: merchants > 2 std deviations above mean failure rate AND with ≥ 50 transactions (statistically stable)
   - Breakdown by failure reason for flagged merchants

2. **Copilot Design**
   - Input: merchant_id → recent failure log (last N transactions, grouped by failure_reason)
   - System prompt: instructs Claude to act as a Paymob support analyst, explain the dominant failure pattern in plain English, and recommend an action (e.g., "Your 3DS failure rate is 42% this week — this typically means your frontend isn't passing the authentication redirect correctly. We recommend testing your 3DS integration in sandbox and opening a ticket if it persists.")
   - Output: plain-English diagnosis + recommended action
   - Evaluation criteria: correctness of diagnosis, actionability of recommendation, absence of hallucinated Paymob-specific details

3. **(Optional) Prototype**
   - Python function: `explain_merchant_failures(merchant_id, df)` → calls Anthropic API → returns explanation string
   - Demo on top 2 anomalous merchants

---

## 5. Deliverables Checklist

| # | Deliverable | File |
|---|-------------|------|
| 1 | Main notebook (all 3 tasks) | `paymob_assessment.ipynb` |
| 2 | Requirements | `requirements.txt` |
| 3 | README with "how to run" | `README.md` |
| 4 | Target customer list (Task 2) | `target_customers_at_risk.csv` |
| 5 | Written summary (~1–2 pages) | Section in README |
| 6 | "If I had more time" note | Section in README |
| 7 | (Bonus) Copilot prototype | `copilot_failures.py` |

---

## 6. Tech Stack

```
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
xgboost>=2.0
imbalanced-learn>=0.12   # SMOTE if used
anthropic>=0.25          # Bonus only
jupyter>=1.0
```

---

## 7. Scoring Alignment

| Rubric Dimension | How We Address It |
|------------------|-------------------|
| **Data handling (20%)** | Section 2 documents every issue and fix explicitly in notebook cells |
| **Analytical depth (20%)** | Ramadan seasonality, peak-hour heatmap, segment-level GMV, per-method failure analysis |
| **Modelling rigour (20%)** | Hard leakage cutoff, stratified CV, AUC primary metric, honest test-set read |
| **Business framing (20%)** | Every finding anchored to a merchant or PM decision; At-Risk segment justified by data |
| **Communication (20%)** | Executive summary in README; notebook tells a linear story with markdown cells |

---

## 8. Timeline Estimate

| Phase | Time |
|-------|------|
| Data cleaning + EDA | 45 min |
| Task 1 (Digest) charts + narrative | 45 min |
| Task 2 (RFM segmentation) | 30 min |
| Task 3 (ML model) | 45 min |
| README + polish | 15 min |
| Bonus (Copilot) | 30 min |
| **Total** | **~3.5 hrs** |

---

## 9. Key Assumptions (to state in notebook)

1. Settled GMV = success rows only. Refunded = money returned, excluded from revenue.
2. Duplicate `transaction_id` rows = export artifact; keep first occurrence.
3. Negative amounts on success rows = data error; drop. Negative amounts on refunded rows = keep as positive (money returned).
4. RFM reference date = 2025-06-30 (last date in data).
5. Campaign leakage cutoff = 2025-04-14 23:59:59 UTC (strictly before campaign date).
6. Outlier transactions (>99.9th percentile) are included in GMV but excluded from segment percentile thresholds.
7. `city` nulls labelled "Unknown" — not imputed.
