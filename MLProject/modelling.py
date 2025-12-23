# ==============================
# FIX WINDOWS UNICODE (WAJIB)
# ==============================
import os
os.environ["DAGSHUB_DISABLE_RICH"] = "1"

# ==============================
# IMPORT
# ==============================
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================
# INIT DAGSHUB + MLFLOW
# ==============================
dagshub.init(
    repo_owner="renal4452",
    repo_name="smsml-mlflow-bank",
    mlflow=True
)

mlflow.set_experiment("bank-deposit-tuning")

# ==============================
# LOAD DATA (PREPROCESSED)
# ==============================
X_train = pd.read_csv("MLProject/bank_preprocessing/X_train.csv")
X_test  = pd.read_csv("MLProject/bank_preprocessing/X_test.csv")
y_train = pd.read_csv("MLProject/bank_preprocessing/y_train.csv").values.ravel()
y_test  = pd.read_csv("MLProject/bank_preprocessing/y_test.csv").values.ravel()

# ==============================
# HYPERPARAMETER GRID
# ==============================
param_grid = {
    "C": [0.01, 0.1, 1.0],
    "solver": ["liblinear"]
}

# ==============================
# TRAINING + LOGGING
# ==============================
for params in ParameterGrid(param_grid):
    with mlflow.start_run():

        model = LogisticRegression(
            max_iter=1000,
            C=params["C"],
            solver=params["solver"]
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ==============================
        # METRICS
        # ==============================
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="yes")
        rec = recall_score(y_test, preds, pos_label="yes")
        f1 = f1_score(y_test, preds, pos_label="yes")

        # ==============================
        # MANUAL LOGGING (WAJIB ADVANCE)
        # ==============================
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", params["C"])
        mlflow.log_param("solver", params["solver"])
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # ==============================
        # LOG MODEL ARTIFACT
        # ==============================
        mlflow.sklearn.log_model(model, "model")

        # ==============================
        # EXTRA ARTIFACT (PLUS POINT)
        # ==============================
        sample_pred = pd.DataFrame({
            "y_true": y_test[:20],
            "y_pred": preds[:20]
        })
        sample_pred.to_csv("prediction_sample.csv", index=False)
        mlflow.log_artifact("prediction_sample.csv")

        print(f"RUN DONE | C={params['C']} | ACC={acc:.4f}")
