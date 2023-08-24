# Importing Dependencies
import logging

import pandas as pd

# Creating a class for data ingestion
class DataIngestion:
    """
    Ingesting from data source and returning a pandas dataframe
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data source
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from the data source

        Returns:
            data: pandas dataframe
        """
        logging.info(f"Ingesting data from {self.data_path}")
        data = pd.read_parquet(self.data_path)
        return data

# Creating a step for data ingestion
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from the data source

    Args:
        data_path: path to the data source

    Returns:
        data: pandas dataframe
    """
    try:
        data = DataIngestion(data_path).get_data()
        logging.info("Data ingestion step completed successfully")
        return data
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e