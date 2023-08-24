# Importing Dependencies
import logging

import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from src.model import CustomLogisticRegressionModel

from typing import Tuple
from typing_extensions import Annotated

# Creating a step for model training
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> Annotated[LogisticRegression, "Trained Model"]:
    """
    Training the model

    Args:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.DataFrame): Training labels
        y_test (pd.DataFrame): Testing labels

    Returns:
        model (Logistic Regression): Trained model
    """
    try:
        model = None
        logging.info("Training the model")

        # Logging the Model Artifacts to MLFlow
        mlflow.sklearn.autolog()

        # Training the model
        model = CustomRandomForestClassifier()
        model.train(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error while training model: {}".format(e))
        raise e