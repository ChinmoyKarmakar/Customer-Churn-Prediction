# Importing Dependencies
import logging
from abc import ABC, abstractmethod

import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create a class for Hyperparameter Tuning Strategy
class HyperparameterTuningStrategy(ABC):
    """
    Abstract class for hyperparameter tuning strategies
    """
    @abstractmethod
    def tune(self, X_train, y_train):
        """
        Abstract method to tune hyperparameters
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        Returns:
            best_params (dict): Best hyperparameters found
        """
        pass

    @abstractmethod
    def get_best_estimator(self):
        """
        Abstract method to return the best estimator from the tuning process
        """
        pass

# Create a class for the Grid Search Strategy
class GridSearchStrategy(HyperparameterTuningStrategy):
    """
    Concrete class for Grid Search hyperparameter tuning
    """
    def __init__(self, model, param_grid):
        """
        Initializes the tuner
        Args:
            model (Model): The model to be tuned
            param_grid (dict): The grid of parameters to search
        """
        self.model = model
        self.param_grid = param_grid
        self.grid_search = None

    def tune(self, X_train, y_train):
        """
        Runs grid search to tune hyperparameters
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        Returns:
            best_params (dict): Best hyperparameters found
        """
        try:
            # Initialize GridSearchCV
            self.grid_search = GridSearchCV(self.model, self.param_grid, cv=5, verbose=1, n_jobs=-1)
            
            # Fit the model with the data
            self.grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = self.grid_search.best_params_
            
            logging.info(f"Best parameters found: {best_params}")
            return best_params
        except Exception as e:
            logging.error(f"Error in hyperparameter tuning: {e}")
            raise e

    def get_best_estimator(self):
        """
        Returns the best estimator from the grid search
        """
        return self.grid_search.best_estimator_