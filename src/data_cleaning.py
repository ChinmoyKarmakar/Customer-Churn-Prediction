# Importing Dependencies
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from typing import Union, Tuple

# Creating a class for data strategy
class DataStrategy(ABC):
    """
    Abstract class for data strategy
    """
    @abstractmethod
    def handle_data(self, *args) -> Union[pd.DataFrame, pd.Series, tuple]:
        """
        Abstract method to handle data
        """
        pass

# Creating a class for data preprocessing strategy
class DataPreprocessingStrategy(DataStrategy):
    """
    Class for data preprocessing strategy
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Drop the unwanted columns
            data = data.drop(['CustomerID', 'Name'], axis=1)

            # Drop the rows with missing values
            data = data.dropna()

            # Label Encoding
            le = LabelEncoder()
            data['Gender'] = le.fit_transform(data['Gender'])
            data['Location'] = le.fit_transform(data['Location'])
            data['Churn'] = le.fit_transform(data['Churn'])

            return data
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e

# Creating a class for data divide strategy
class DataDivideStrategy(DataStrategy):
    """
    Class for data feature engineering strategy
    """
    def handle_data(self, data: pd.DataFrame) -> tuple:
        """
        Divide the data into train and test
        """
        try:
            # Drop the target variable
            X = data.drop('Churn', axis=1)
            y = data['Churn']

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
                        
# Creating a class for data feature engineering strategy
class DataFeatureEngineeringStrategy(DataStrategy):
    """
    Class for data feature engineering strategy
    """
    def handle_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple:
        """
        Feature engineering
        """
        try:
            # Feature Extraction
            X_train['Avg_Usage_per_Month'] = X_train['Total_Usage_GB'] / X_train['Subscription_Length_Months']
            X_test['Avg_Usage_per_Month'] = X_test['Total_Usage_GB'] / X_test['Subscription_Length_Months']

            X_train['Bill_to_Usage_Ratio'] = X_train['Monthly_Bill'] / X_train['Total_Usage_GB']
            X_test['Bill_to_Usage_Ratio'] = X_test['Monthly_Bill'] / X_test['Total_Usage_GB']

            X_train['Age_to_Usage_Ratio'] = X_train['Age'] / X_train['Total_Usage_GB']
            X_test['Age_to_Usage_Ratio'] = X_test['Age'] / X_test['Total_Usage_GB']

            # Features to be scaled
            features = [col for col in X_train.columns if col not in ['Churn']]

            # Initialize the scaler
            scaler = StandardScaler()
            X_train[features] = scaler.fit_transform(X_train[features])
            X_test[features] = scaler.transform(X_test[features])
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise e

# Creating a class for data strategy context
class DataStrategyContext:
    """
    Class for data strategy context
    """
    def __init__(self, strategy: DataStrategy, *args) -> None:
        """
        Initialize the data and strategy
        """
        self.data = args
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.Series]]:
        """
        Handle the data
        """
        try:
            return self.strategy.handle_data(*self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e