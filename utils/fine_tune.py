# Importing Dependencies
import logging

import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from src.model import CustomLogisticRegressionModel
from src.fine_tuning import HyperparameterTuningStrategy, GridSearchStrategy

from typing import Tuple, Union
from typing_extensions import Annotated

# Creating a step for model training with hyperparameter tuning
def train_and_tune_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> Annotated[Union[LogisticRegression, CustomLogisticRegressionModel], "Trained Model"]:
    """
    Training and tuning the model

    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels

    Returns:
        model (Logistic Regression or CustomLogisticRegressionModel): Trained and tuned model
    """
    try:
        logging.info("Training and tuning the model")

        # Define the hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        tuner = GridSearchStrategy(logistic_model, param_grid)

        # Hyperparameter tuning
        best_params = tuner.tune(X_train, y_train)
        logging.info(f"Best hyperparameters: {best_params}")

        # Logging the Model Artifacts to MLFlow
        mlflow.sklearn.autolog()

        # Training the model with best hyperparameters
        model = CustomLogisticRegressionModel(hyperparameters=best_params)
        model.train(X_train, y_train)

        return model
    except Exception as e:
        logging.error("Error while training and tuning model: {}".format(e))
        raise e