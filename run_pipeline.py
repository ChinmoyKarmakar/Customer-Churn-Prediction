# Importing Dependencies
import logging

from pipeline.train_pipeline import train_pipeline

# Defining the main function
if __name__ == '__main__':
    train_pipeline(data_path="data/customer_churn_large_dataset_optimized.parquet")