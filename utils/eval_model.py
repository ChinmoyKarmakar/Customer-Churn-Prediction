# Importing Dependencies
import logging

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import Accuracy, Precision, Recall, F1Score

from typing import Tuple
from typing_extensions import Annotated

# Creating a step for model evaluation
def eval_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "Accuracy"],
    Annotated[float, "Precision"],
    Annotated[float, "Recall"],
    Annotated[float, "F1 Score"]]:
    """
    Evaluating the model on the dataframe
    Args:
        model (ClassifierMixin): Trained model
        X_test (pd.DataFrame): Testing data
        y_test (pd.DataFrame): Testing labels
    Returns:
        Accuracy (float): Accuracy
        Precision (float): Precision
        Recall (float): Recall
        F1 Score (float): F1 Score
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

        logging.info(f"Accuracy Score: {acc}")
        logging.info(f"Precision Score: {prec}")
        logging.info(f"Recall Score: {rec}")
        logging.info(f"F1 Score: {f1}")

        return acc, prec, rec, f1
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}")
        raise e