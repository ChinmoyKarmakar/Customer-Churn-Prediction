# Importing Dependencies
import logging

from utils.ingest_data import ingest_data
from utils.clean_data import clean_data
from utils.train_model import train_model
from utils.eval_model import eval_model
from utils.fine_tune import train_and_tune_model

# Defining the Training Pipeline
def train_pipeline(data_path: str):
    """
    Training pipeline to train the model
    Args:
        data_path: str, path to the data
    Returns:
        X_train, X_test, y_train, y_test: Training and testing data
    """
    df = ingest_data(data_path)
    print("Data Ingested Successfully")
    
    X_train, X_test, y_train, y_test = clean_data(df)
    print("Data Preprocessed Successfully")
    
    model = train_model(X_train, y_train)
    print("Logistic Regression Model Trained Successfully")

    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score = eval_model(model, X_test, y_test)
    print("Model Evaluated Successfully")
    
    print("Metrics before Hyperparameter Tuning:")
    print(f"Accuracy: {accuracy_score}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1 Score: {f1_score}")
    print(f"ROC AUC Score: {roc_auc_score}")

    return X_train, X_test, y_train, y_test