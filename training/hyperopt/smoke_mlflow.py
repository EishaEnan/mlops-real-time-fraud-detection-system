# training/hyperopt/smoke_mlflow.py
from __future__ import annotations

import json
import os
import random

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT", "smoke_fast")
ARTIFACTS_URI = os.getenv("ARTIFACTS_URI", "").rstrip("/")
EXP_ARTIFACT_DIR = os.getenv("EXP_ARTIFACT_DIR", EXP_NAME)


def ensure_experiment(name: str) -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    artifact_location = f"{ARTIFACTS_URI}/{EXP_ARTIFACT_DIR}" if ARTIFACTS_URI else None
    return client.create_experiment(name, artifact_location=artifact_location)


def main():
    mlflow.set_tracking_uri(TRACKING)
    exp_id = ensure_experiment(EXP_NAME)
    mlflow.set_experiment(EXP_NAME)  # binds following runs to this experiment

    # tiny dataset so it’s instant
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    with mlflow.start_run(run_name="smoke"):
        # train a super-fast model
        model = LogisticRegression(max_iter=200, n_jobs=1)
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))

        # log basics
        mlflow.log_param("algo", "logreg")
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(model, "model")  # saved under the run's artifacts

        # log a tiny artifact
        with open("summary.json", "w") as f:
            json.dump({"accuracy": float(acc)}, f)
        mlflow.log_artifact("summary.json")

        # a couple of nested "trials" to mimic hyperopt structure
        for depth in [2, 3, 4]:
            with mlflow.start_run(run_name=f"trial_depth={depth}", nested=True):
                mlflow.log_param("depth", depth)
                mlflow.log_metric("score", random.random())

        run_id = mlflow.active_run().info.run_id

    print(f"Run: {run_id}")
    print(f"UI:  {TRACKING}/#/experiments/{exp_id}/runs/{run_id}")


if __name__ == "__main__":
    main()
