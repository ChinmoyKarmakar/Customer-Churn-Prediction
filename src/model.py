# Importing Dependencies
import logging
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier

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

# Create a class for the Random Forest Model
class CustomRandomForestClassifier(Model):
    """
    Class for the Custom Random Forest Model
    """
    def __init__(self):
        """
        Initializes the model
        """
        self.rf = None

    def train(self, X_train, y_train) -> RandomForestClassifier:
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        Returns:
            rf: Random Forest Model
        """
        try:
            # Train the model
            self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf.fit(X_train, y_train)

            logging.info("Training completed successfully")
            return self.rf
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e

    def predict(self, X_test):
        """
        Predicts the values
        Args:
            model: Trained model
            X_test (pd.DataFrame): Testing data
        Returns:
            y_pred: Predicted values
        """
        try:
            # Predict the values
            y_pred = self.rf.predict(X_test)

            logging.info("Prediction completed successfully")
            return y_pred
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise e