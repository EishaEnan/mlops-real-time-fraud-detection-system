# 🧠 Model Training Summary — PaySim Fraud Detection

## 1. Goal
Develop and evaluate fraud detection models using the PaySim dataset, tackling heavy class imbalance and complex non-linear patterns.

---

## 2. Models Tried

| Model              | ROC AUC       | Avg Precision | Recall (Fraud) | Notes                                 |
|-------------------|---------------|----------------|----------------|---------------------------------------|
| Logistic Regression | 0.9563       | 0.5857         | 0.96           | Good baseline, but poor fraud precision |
| Random Forest       | 0.9996       | 0.9612         | 0.92           | Overfit tendencies; slow to train      |
| XGBoost             | 0.9993       | 0.9522         | 0.94           | ⚡️ Fast, high-performing, robust       |
| Neural Network      | 0.4991       | 0.0013         | 0.00           | Failed to learn fraud signal           |

✅ **XGBoost was selected** for final tuning due to its:
- Excellent trade-off between speed and performance
- High recall and precision on fraud class
- Strong generalization compared to overfit-prone Random Forest

---

## 3. XGBoost Hyperparameter Optimization

Performed using `Hyperopt` and `MLflow` tracking:
- Search space: `max_depth`, `learning_rate`, `n_estimators`, `gamma`, `subsample`, `colsample_bytree`
- Evaluation metric: **Validation Average Precision (AP)**
- Trials: 20

### 🔍 Best Parameters Found:

```python
{
    'max_depth': 9,
    'learning_rate': 0.0642,
    'n_estimators': 160,
    'gamma': 1.9331,
    'subsample': 0.8042,
    'colsample_bytree': 0.7916
}
```
---

## 4\. Next Steps 
 ➡️ Proceed with: 
 * Final training using best XGBoost parameters 
 * Evaluation on test set 
 * Model explainability (SHAP) 
 * Model registration and deployment with FastAPI 
 
📓 Notebook: `04_model_xgboost.ipynb` 🧪 Experiment tracking: `MLflow` (local or remote)”

