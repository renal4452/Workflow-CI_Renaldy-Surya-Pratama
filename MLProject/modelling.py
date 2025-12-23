import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
X_train = pd.read_csv("MLProject/bank_preprocessing/X_train.csv")
X_test  = pd.read_csv("MLProject/bank_preprocessing/X_test.csv")
y_train = pd.read_csv("MLProject/bank_preprocessing/y_train.csv").values.ravel()
y_test  = pd.read_csv("MLProject/bank_preprocessing/y_test.csv").values.ravel()

mlflow.set_experiment("bank-deposit-tuning")

param_grid = {
    "C": [0.01, 0.1, 1.0],
    "solver": ["liblinear"]
}

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        model = LogisticRegression(
            max_iter=1000,
            C=params["C"],
            solver=params["solver"]
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="yes")
        rec = recall_score(y_test, preds, pos_label="yes")
        f1 = f1_score(y_test, preds, pos_label="yes")

        # Manual logging
        mlflow.log_param("C", params["C"])
        mlflow.log_param("solver", params["solver"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")
