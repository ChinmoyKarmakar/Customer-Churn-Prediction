# Importing Dependencies
import logging

from pipeline.train_pipeline import train_pipeline
from pipeline.finetune_pipeline import fine_tune_pipeline

# Defining the main function
if __name__ == '__main__':
    # Setting up the logger
    logging.basicConfig(level=logging.INFO, filename="logs.txt")
    logging.info("Starting the Training Pipeline")

    data_path="data/customer_churn_large_dataset_optimized.parquet"
    X_train, X_test, y_train, y_test = train_pipeline(data_path)
    print("Training Pipeline Completed Successfully")

    logging.info("Training Pipeline Completed Successfully")

    model = fine_tune_pipeline(X_train, y_train, X_test, y_test)
    print("Fine Tuning Pipeline Completed Successfully")

    logging.info("Fine Tuning Pipeline Completed Successfully")

    # Saving the model
    save_path = "saved_models/logistic_regression_model.pkl"
    model.save(save_path)
    print("Model Saved Successfully")

    logging.info("Model Saved Successfully")