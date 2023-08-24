# Importing Dependencies
import logging

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from src.model import CustomRandomForestClassifier

from typing import Tuple
from typing_extensions import Annotated

# Creating a step for model training
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> Annotated[RandomForestClassifier, "Trained Model"]:
    """
    Training the model

    Args:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.DataFrame): Training labels
        y_test (pd.DataFrame): Testing labels

    Returns:
        model (ClassifierMixin): Trained model
    """
    try:
        model = None
        model = CustomRandomForestClassifier()
        model.train(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error while training model: {}".format(e))
        raise e