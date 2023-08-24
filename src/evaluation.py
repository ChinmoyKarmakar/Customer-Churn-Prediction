# Importing Dependencies
import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Creating a class for model evaluation
class ModelEvaluation(ABC):
    """
    Abstract class for model evaluation
    """
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        Returns:
            None
        """
        pass

# Creating a class for Accuracy
class Accuracy(ModelEvaluation):
    """
    Evaluation Strategy for Accuracy
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy")
            acc = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy: {acc}")
            return acc
        except Exception as e:
            logging.error(f"Error in calculating Accuracy: {e}")
            raise e

# Creating a class for Precision
class Precision(ModelEvaluation):
    """
    Evaluation Strategy for Precision
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Precision")
            prec = precision_score(y_true, y_pred)
            logging.info(f"Precision: {prec}")
            return prec
        except Exception as e:
            logging.error(f"Error in calculating Precision: {e}")
            raise e

# Creating a class for Recall
class Recall(ModelEvaluation):
    """
    Evaluation Strategy for Recall
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Recall")
            rec = recall_score(y_true, y_pred)
            logging.info(f"Recall: {rec}")
            return rec
        except Exception as e:
            logging.error(f"Error in calculating Recall: {e}")
            raise e

# Creating a class for F1 Score
class F1Score(ModelEvaluation):
    """
    Evaluation Strategy for F1 Score
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1 Score")
            f1 = f1_score(y_true, y_pred)
            logging.info(f"F1 Score: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Error in calculating F1 Score: {e}")
            raise e