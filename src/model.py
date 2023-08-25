# Importing Dependencies
import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# Create a class for the Model
class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        Returns:
            model: Trained model
        """
        pass

    @abstractmethod 
    def predict(self, X_test):
        """
        Predicts the values
        Args:
            X_test (pd.DataFrame): Testing data
        Returns:
            y_pred: Predicted values
        """
        pass

    @abstractmethod
    def save(self, filename: str):
        """
        Save the model to a file
        Args:
            filename (str): Path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filename: str):
        """
        Load the model from a file
        Args:
            filename (str): Path to load the model from
        Returns:
            model_instance: An instance of this class with the loaded model
        """
        pass

    @abstractmethod
    def predict_proba(self, X_test):
        """
        Predicts class probabilities
        Args:
            X_test (pd.DataFrame): Testing data
        Returns:
            y_proba: Predicted class probabilities
        """
        pass

# Create a class for the Logistic Regression Model
class CustomLogisticRegressionModel(Model):
    """
    Class for the Custom Logistic Regression Model
    """
    def __init__(self, hyperparameters=None):
        """
        Initializes the model
        Args:
            hyperparameters (dict, optional): Dictionary of hyperparameters
        """
        self.lr = None
        self.hyperparameters = hyperparameters if hyperparameters else {}

    def train(self, X_train, y_train) -> LogisticRegression:
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        Returns:
            lr: Logistic Regression Model
        """
        try:
            # Train the model with specified hyperparameters
            self.lr = LogisticRegression(random_state=42, max_iter=1000, **self.hyperparameters)
            self.lr.fit(X_train, y_train)

            logging.info("Training completed successfully")
            return self.lr
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e

    def predict(self, X_test):
        """
        Predicts the values
        Args:
            X_test (pd.DataFrame): Testing data
        Returns:
            y_pred: Predicted values
        """
        try:
            # Predict the values
            y_pred = self.lr.predict(X_test)

            logging.info("Prediction completed successfully")
            return y_pred
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise e

    def save(self, filename: str):
        """
        Save the model to a file
        Args:
            filename (str): Path to save the model
        """
        try:
            dump(self.lr, filename)
            logging.info(f"Model saved successfully to {filename}")
        except Exception as e:
            logging.error(f"Error saving the model: {e}")
            raise e

    @classmethod
    def load(cls, filename: str):
        """
        Load the model from a file
        Args:
            filename (str): Path to load the model from
        Returns:
            model_instance: An instance of this class with the loaded model
        """
        try:
            loaded_model = load(filename)
            instance = cls()
            instance.lr = loaded_model
            logging.info(f"Model loaded successfully from {filename}")
            return instance
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            raise e

    def predict_proba(self, X_test):
        """
        Predicts class probabilities
        Args:
            X_test (pd.DataFrame): Testing data
        Returns:
            y_proba: Predicted class probabilities
        """
        try:
            # Predict class probabilities
            y_proba = self.lr.predict_proba(X_test)

            logging.info("Probability prediction completed successfully")
            return y_proba
        except Exception as e:
            logging.error(f"Error in probability prediction: {e}")
            raise e