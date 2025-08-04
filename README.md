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

🔗 **Live App (Coming Soon):** [https://ml-tfds.eishaenan.com](https://ml-tfds.eishaenan.com)

## 📝 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for more details.