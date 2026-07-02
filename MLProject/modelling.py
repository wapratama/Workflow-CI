"""
modelling.py (MLProject version)

Versi modelling.py yang disesuaikan untuk dijalankan sebagai MLflow Project
melalui `mlflow run`, dipicu otomatis oleh GitHub Actions CI (Kriteria 3).

Perbedaan dengan modelling.py di Kriteria 2:
- Menerima parameter via command-line (argparse), sesuai definisi di file MLProject
- Menggunakan local file-based tracking (./mlruns) -- tidak butuh server terpisah
- Hyperparameter sudah merupakan hasil terbaik dari tuning di Kriteria 2
  (n_estimators=200, max_depth=25, class_weight='balanced'), dengan default
  yang bisa dioverride untuk fleksibilitas re-training
- Mencetak run_id ke stdout dalam format yang mudah di-parse oleh workflow CI
- Experiment ditentukan lewat flag --experiment-name pada command `mlflow run`
  itu sendiri, BUKAN di dalam script -- karena `mlflow run` CLI sudah membuat
  run context sendiri, memanggil mlflow.set_experiment() di sini akan
  menyebabkan experiment_id mismatch dengan run yang sudah dibuat CLI.

Dijalankan via:
    uv run mlflow run . --env-manager=local \
        --experiment-name "Facies Classification - CI Retraining" \
        -P data_path=facies_preprocessing.csv \
        -P n_estimators=200 \
        -P max_depth=25
"""

import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


RANDOM_STATE = 42
TARGET_COL   = "Facies"


def parse_args():
    parser = argparse.ArgumentParser(description="Training Facies Classification model (MLProject)")
    parser.add_argument("--data_path", type=str, default="facies_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=25)
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset dimuat: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy"        : accuracy_score(y_true, y_pred),
        "f1_macro"        : f1_score(y_true, y_pred, average="macro"),
        "f1_weighted"     : f1_score(y_true, y_pred, average="weighted"),
        "precision_macro" : precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro"    : recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    args = parse_args()

    # Tracking lokal (file-based ./mlruns) -- akan di-commit oleh workflow CI.
    # PENTING: saat dijalankan via `mlflow run`, MLflow CLI sudah membuat run
    # di dalam experiment context yang ditentukan lewat flag --experiment-name
    # pada command `mlflow run` itu sendiri.
    # Jika kita memanggil mlflow.set_experiment() di sini SETELAH run dibuat,
    # itu akan menyebabkan experiment_id mismatch dengan run yang sudah ada.
    # Maka set_experiment() hanya dipanggil saat TIDAK ada run aktif dari env
    # (yaitu saat script dijalankan langsung dengan `python modelling.py`,
    # bukan lewat `mlflow run`).
    running_as_mlproject = os.environ.get("MLFLOW_RUN_ID") is not None

    if not running_as_mlproject:
        mlflow.set_experiment("Facies Classification - CI Retraining")

    mlflow.sklearn.autolog()

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    started_here = mlflow.active_run() is None
    if started_here:
        env_run_id = os.environ.get("MLFLOW_RUN_ID")
        mlflow.start_run(run_id=env_run_id) if env_run_id else mlflow.start_run()

    run = mlflow.active_run()

    model = RandomForestClassifier(
        n_estimators = args.n_estimators,
        max_depth     = args.max_depth,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1
    )
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)

    for name, value in test_metrics.items():
        mlflow.log_metric(f"test_{name}", value)

    print(f"\nTest accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test f1_macro : {test_metrics['f1_macro']:.4f}")

    run_id = run.info.run_id
    # Format khusus agar mudah di-parse oleh workflow CI (grep/cut)
    print(f"\nMLFLOW_RUN_ID={run_id}")
    print(f"Training selesai. Run ID: {run_id}")

    if started_here:
        mlflow.end_run()


if __name__ == "__main__":
    main()
