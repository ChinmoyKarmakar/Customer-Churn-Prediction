# Importing Dependencies
import logging

import pandas as pd

from src.data_cleaning  import DataPreprocessingStrategy, DataDivideStrategy, DataFeatureEngineeringStrategy, DataStrategyContext
from typing import Tuple
from typing_extensions import Annotated

# Creating a step for data cleaning
# Importing Dependencies
import logging

import pandas as pd

from src.data_cleaning  import DataPreprocessingStrategy, DataDivideStrategy, DataFeatureEngineeringStrategy, DataStrategyContext
from typing import Tuple
from typing_extensions import Annotated

# Creating a step for data cleaning
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans the data, performs feature extraction and scaling and splits it into train and test sets

    Args:
        df: Input dataframe
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataStrategyContext(process_strategy, df)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_divide = DataStrategyContext(divide_strategy, processed_data)
        X_train, X_test, y_train, y_test = data_divide.handle_data()

        feature_strategy = DataFeatureEngineeringStrategy()
        data_feature = DataStrategyContext(feature_strategy, X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = data_feature.handle_data()

        logging.info("Data cleaning step completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Data cleaning step failed with error {e}")
        raise e