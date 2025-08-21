# 🛡️ Real-Time MLOps Fraud Detection System

This project is a real-time, end-to-end MLOps platform for detecting fraudulent transactions. It integrates model training, experiment tracking, API deployment, and dashboarding in a modular and reproducible way.

## 🚀 Key Features

- Real-time fraud prediction via FastAPI
- Interactive dashboard using Streamlit
- Automated training with hyperparameter tuning (Hyperopt)
- Experiment tracking with MLflow
- Modular pipeline (data → train → deploy → monitor)
- Designed for local (Docker) and cloud (EC2) deployment

## 📦 Tech Stack

**MLOps:** MLflow, DVC, Airflow  
**Serving:** FastAPI  
**UI:** Streamlit  
**Modeling:** scikit-learn, XGBoost, Hyperopt  
**Infra:** Docker, AWS EC2

---
### 📂 Project Structure

- `src/` – Core pipeline logic (data ingestion, preprocessing, training, evaluation, etc.)
- `data/` – Raw and processed datasets (tracked with DVC)
- `notebooks/` – EDA and prototyping notebooks
- `models/` – Trained models (also tracked with DVC)
- `streamlit_app/` – Frontend UI using Streamlit
- `scripts/` – Utility or automation scripts

---
## 📂 Dataset

### Source: Synthetic Financial Datasets for Fraud Detection

This dataset is a synthetic representation of mobile money transactions, generated using the **PaySim** simulator based on real transaction data from a financial service provider in an African country.

- 📦 **File**: `data/raw/paysim.csv`
- 📈 **Rows**: ~6 million (1 month of transactions, 744 steps = 744 hours)
- 💳 **Transaction Types**: PAYMENT, TRANSFER, DEBIT, CASH_IN, CASH_OUT
- ⚠️ **Fraud Indicator**: `isFraud` (target), `isFlaggedFraud` (flagged high-risk ops)
- 🧾 **Note**: Balances (`oldbalanceOrig`, etc.) should not be used for fraud detection — they are post-processed to reflect canceled fraud transactions.

📊 **[EDA Summary](./reports/eda_summary.md)** — key findings and insights from data exploration.


📚 **Original Source**: [Kaggle – PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)  
📄 **Citation**:
> Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). _PaySim: A financial mobile money simulator for fraud detection_. The 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus.

📘 **Related Research**: [PhD Thesis](http://urn.kb.se/resolve?urn=urn:nbn:se:bth-12932)

### FastAPI Model Serving

Once a model version is registered in MLflow, launch the API with:

```bash
uvicorn mlops_fraud.deployment.api:app --host 0.0.0.0 --port 8000
```

The service reads `MLFLOW_TRACKING_URI`, `MODEL_NAME` and `MODEL_ALIAS` to locate the model.
Send a POST request to `/predict` with a JSON body matching `schemas.TransactionRequest`.

---

🔗 **Live App (Coming Soon):** [https://ml-tfds.eishaenan.com](https://ml-tfds.eishaenan.com)

## 📝 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for more details.