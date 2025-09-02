# airflow/dags/fraud_train_daily.py
from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/project"
RUN_DIR = PROJECT_ROOT + "/reports/runs/{{ ds }}"

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="fraud_train_daily",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,           # change to '@daily' later
    catchup=False,
    tags=["mlops", "fraud_detection"],
    default_args=default_args,
) as dag:

    BASE_ENV = {
        "PYTHONPATH": "/opt/project/src",
        "MLFLOW_TRACKING_URI": "{{ var.value.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000') }}",
        "ARTIFACTS_URI": "{{ var.value.get('ARTIFACTS_URI', 's3://mlops-fraud-dvc') }}",
        "MLFLOW_MODEL_NAME": "{{ var.value.get('MLFLOW_MODEL_NAME', 'fraud_xgb') }}",
        "LABEL_COL": "{{ var.value.get('LABEL_COL', 'isFraud') }}",
        "METRIC_NAME": "{{ var.value.get('METRIC_NAME', 'pr_auc') }}",
        "PROMOTE_THRESHOLD": "{{ var.value.get('PROMOTE_THRESHOLD', '0.85') }}",
        "AWS_REGION": "{{ var.value.get('AWS_REGION', '') }}",
        "AWS_ACCESS_KEY_ID": "{{ var.value.get('AWS_ACCESS_KEY_ID', '') }}",
        "AWS_SECRET_ACCESS_KEY": "{{ var.value.get('AWS_SECRET_ACCESS_KEY', '') }}",
    }

    split_data = BashOperator(
        task_id="dvc_split_data",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project && /home/airflow/.local/bin/dvc repro split_data --force"
        ),
        env=BASE_ENV,
        dag=dag,
    )

    # hyperopt = BashOperator(
    #     task_id="hyperopt",
    #     bash_command=(
    #         "set -euo pipefail; "
    #         "cd /opt/project && PYTHONPATH=/opt/project/src python training/hyperopt/search_xgb.py"
    #     ),
    #     env=dict(BASE_ENV, RUN_DIR=RUN_DIR),
    #     dag=dag,
    # )

    params_fixed = BashOperator(
        task_id="params_fixed",
        bash_command=(
            "set -euo pipefail; "
            f"mkdir -p {RUN_DIR} && cat > {RUN_DIR}/best_xgb_params.json <<'JSON'\n"
            '{"colsample_bytree": 0.877278091860128, "gamma": 0.0007412820161393726, '
            '"learning_rate": 0.012951232741934696, "max_depth": 4, '
            '"min_child_weight": 1.1305546123305603, "n_estimators": 1200, '
            '"subsample": 0.7356029819453461}\n'
            "JSON"
        ),
        env=BASE_ENV,
        dag=dag,
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project && PYTHONPATH=/opt/project/src python training/pipelines/train_xgb.py"
        ),
        env=dict(BASE_ENV, RUN_DIR=RUN_DIR),
        dag=dag,
    )

    evaluate = BashOperator(
        task_id="eval",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project && PYTHONPATH=/opt/project/src python training/pipelines/evaluate.py"
        ),
        env=dict(BASE_ENV, RUN_DIR=RUN_DIR),
        dag=dag,
    )

    promote = BashOperator(
        task_id="promote",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project && PYTHONPATH=/opt/project/src python scripts/promote_best.py"
        ),
        env=dict(BASE_ENV, RUN_DIR=RUN_DIR),
        dag=dag,
    )

    # Pick one path. For first green run, keep fixed params:
    split_data >> params_fixed >> train >> evaluate >> promote
    # split_data >> hyperopt >> train >> evaluate >> promote
