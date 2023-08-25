# Customer Churn Prediction

## Introduction
The repository contains the code for the Customer Churn Prediction project. The project is divided into two parts:
1. Machine Learning Pipeline
2. Web Application

The Machine Learning Pipeline is used to train the model and save it in a pickle file. The Web Application is used to deploy the model and make predictions.

## Setup
1. Clone the repo
    ```
    git clone https://github.com/suryanshgupta9933/Customer-Churn-Prediction.git
    ```
2. Create a virtual environment
    ```
    python -m venv env
    ```
3. Activate the virtual environment
    ```
    env\Scripts\activate
    ```
4. Install the dependencies
    ```
    pip install -r requirements.txt
    ```

## Machine Learning Pipeline

### 1. Data Collection and Optimization.
- The dataset is provided in the repository as an excel file under `data` folder.
- The data is first optimized by changing the data types of the columns to reduce the memory usage.
- The excel file is converted to a parquet file to reduce the file size.

### 2. Data Preprocessing.
- There were no missing values in the dataset.
- There were no outliers in the dataset.
- The categorical columns were encoded using Label Encoder.
- The dataset was split into train and test sets in the ratio 80:20.

> Note: There was very low correlation between the features and the target variable. This is the reason why the models are not performing well.

### 3. Feature Engineering and Scaling.
- Three new features were created:
    - `Avg_Usage_per_Month` = `Total_Usage_GB` / `Subscription_Length_Months`
    - `Bill_to_Usage_Ratio` = `Monthly_Bill` / `Total_Usage_GB`
    - `Age_to_Usage_Ratio` = `Age` / `Total_Usage_GB`
- The distribution of target variable was checked and it was already balanced.
- The features were scaled using Standard Scaler.

### 4. Model Training.
- The following models were trained:
    - Logistic Regression
    - Random Forest Classifier
    - CatBoost Classifier
    - LightGBM Classifier
- Each model showed a very similar performance with an accuracy of around 50%.
- I went ahead with the Logistic Regression model in the pipeline as it was the simplest model and was also performing well.

> Note: The dataset is probably synthetic and not the actual representation of the real world data. This is the reason why the models are not performing well.

### 5. Hyperparameter Tuning.
- The hyperparameters were tuned using Grid Search CV.
- The best parameters were used to train the model again.
