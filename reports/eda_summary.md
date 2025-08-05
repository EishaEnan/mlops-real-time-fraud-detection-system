# EDA Summary — PaySim Dataset

## 1. Dataset Overview
- Total transactions: 6,362,620
- Fraudulent transactions: 8,213 (~0.13%)
- Class imbalance: Heavy skew toward non-fraudulent data

## 2. Transaction Type
- Transaction types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- Fraud only occurs in: **TRANSFER** and **CASH_OUT**
- Highest fraud rate: TRANSFER (~0.78%), CASH_OUT (~0.18%)

## 3. Amount Analysis
- Amount is highly right-skewed
- Fraudulent transactions tend to involve **higher amounts**
- Feature: `log_amount = log1p(amount)` is recommended

## 4. Balance Logic Checks
- Created:
  - `errorOrig = oldbalanceOrg - amount - newbalanceOrig`
  - `errorDest = oldbalanceDest + amount - newbalanceDest`
- These revealed mismatches — useful for feature engineering
- Also created: `abs_errorOrig`, `abs_errorDest`

## 5. isFlaggedFraud
- Only 16 transactions were flagged
- All 16 were actual frauds ⇒ 100% precision, but recall ≈ 0.2%
- Useful, but limited rule-based signal

## 6. Temporal Trends
- Time steps span 744 hours (~31 days)
- Fraud distributed fairly evenly across time
- Created `hour = step % 24`, `day = step // 24` for feature use

## 7. Correlation Analysis
- High redundancy between:
  - `oldbalanceOrg` & `newbalanceOrig` (corr ~1)
  - `oldbalanceDest` & `newbalanceDest` (corr ~0.98)
- Weak correlation between any individual feature and `isFraud`
- Suggests non-linear modeling will be needed

---

## Next Steps

➡️ Move to `02_feature_engineering.ipynb`:
- Drop/transform redundant features
- Create `deltaOrig`, `deltaDest`, `log_amount`, time features
- Encode `type` feature
- Prepare for class imbalance handling & modeling

