# Importing Dependencies
import logging

from utils.ingest_data import ingest_data
from utils.clean_data import clean_data
from utils.train_model import train_model
from utils.eval_model import eval_model

# Defining the Training Pipeline
def train_pipeline(data_path: str):
    """
    Training pipeline to train the model
    Args:
        data_path: str, path to the data
    Returns:
        None
    """
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score = eval_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy_score}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1 Score: {f1_score}")
