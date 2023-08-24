# Importing Dependencies
import logging

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import Accuracy, Precision, Recall, F1Score, ROCAUCScore

from typing import Tuple
from typing_extensions import Annotated

# Creating a step for model evaluation
def eval_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "Accuracy"],
    Annotated[float, "Precision"],
    Annotated[float, "Recall"],
    Annotated[float, "F1 Score"],
    Annotated[float, "ROC AUC Score"]
]:
    """
    Evaluating the model on the dataframe
    Args:
        model (Logistic Regression): Trained model
        X_test (pd.DataFrame): Testing data
        y_test (pd.DataFrame): Testing labels
    Returns:
        Accuracy (float): Accuracy
        Precision (float): Precision
        Recall (float): Recall
        F1 Score (float): F1 Score
        ROC AUC Score (float): ROC AUC Score
    """
    try:
        logging.info("Evaluating the model")

        # Predicting the values
        y_pred = model.predict(X_test)

        # Calculating the metrics
        acc = Accuracy().calculate(y_test, y_pred)
        prec = Precision().calculate(y_test, y_pred)
        rec = Recall().calculate(y_test, y_pred)
        f1 = F1Score().calculate(y_test, y_pred)
        rocauc = ROCAUCScore().calculate(y_test, y_pred)

        # Logging the metrics to MLFlow
        mlflow.log_metrics({
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC Score": rocauc
        })

        logging.info("Evaluation completed successfully")

        return acc, prec, rec, f1, rocauc
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}")
        raise e