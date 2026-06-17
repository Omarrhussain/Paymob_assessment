# Data Dictionary

You are given two CSV files in this folder. They are a **raw export** from a
synthetic payments environment - treat them the way you would treat real
production data (i.e. do not assume they are clean).

All data is **synthetic**. No real customers, merchants, or transactions are
represented.

---

## `transactions.csv`

One row per payment attempt.

| Column            | Type     | Description |
|-------------------|----------|-------------|
| `transaction_id`  | string   | Unique transaction identifier (e.g. `T00012345`). |
| `customer_id`     | string   | The paying customer (e.g. `C01234`). Joins to `campaign_responses.csv`. |
| `merchant_id`     | string   | The merchant receiving the payment (e.g. `M042`). |
| `mcc_category`    | string   | Merchant category (Groceries, Restaurants, Fashion, Electronics, Pharmacy, Travel, Beauty, Home & Furniture, Telecom, Entertainment). |
| `timestamp_utc`   | datetime | When the attempt happened, in **UTC** (`YYYY-MM-DD HH:MM:SS`). Note: Paymob operates in Egypt - local time is **Africa/Cairo (UTC+2)**. |
| `amount_egp`      | float    | Transaction amount in Egyptian Pounds. |
| `payment_method`  | string   | One of: wallet, card, instapay, cash_on_delivery, installment. |
| `channel`         | string   | `online` or `pos`. |
| `status`          | string   | `success`, `failed`, or `refunded`. |
| `failure_reason`  | string   | Populated when `status = failed` (e.g. `insufficient_funds`, `integration_error`, `3ds_authentication_failed`, `timeout`, ...). Empty otherwise. |
| `city`            | string   | Customer governorate. |

Notes:
- The export spans roughly **May 2024 - June 2025**.
- Revenue/GMV should reflect money that actually settled - think about which
  `status` values count.
- As with any production export, expect some data-quality issues.

---

## `campaign_responses.csv`

The outcome of one past marketing campaign. Only customers who were **targeted**
by the campaign appear here.

| Column          | Type    | Description |
|-----------------|---------|-------------|
| `customer_id`   | string  | Targeted customer. Joins to `transactions.csv`. |
| `campaign_name` | string  | Campaign label (`Q2 High-Value Win-Back`). |
| `campaign_date` | date    | The date the campaign was sent: **2025-04-15**. |
| `responded`     | int     | `1` if the customer made a qualifying purchase after the campaign, else `0`. |

Note: the `campaign_date` matters a great deal if you intend to use this table
as a modelling target.
