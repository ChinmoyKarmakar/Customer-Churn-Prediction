# Importing Dependencies
import logging

from utils.eval_model import eval_model
from utils.fine_tune import train_and_tune_model

# Defining the Fine Tuning Pipeline
def fine_tune_pipeline(X_train, y_train, X_test, y_test):
    """
    Fine tuning pipeline to fine tune the model
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
    Returns:
        model: Fine tuned model
    """
    tuned_model = train_and_tune_model(X_train, y_train)
    print("Logistic Regression Model Trained and Tuned Successfully")

    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score = eval_model(tuned_model, X_test, y_test)
    print("Model Evaluated Successfully")

    print("Metrics after Hyperparameter Tuning:")
    print(f"Accuracy: {accuracy_score}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1 Score: {f1_score}")
    print(f"ROC AUC Score: {roc_auc_score}")