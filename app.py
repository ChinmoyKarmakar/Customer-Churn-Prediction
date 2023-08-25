# Importing Dependencies
import streamlit as st
import pandas as pd
from src.model import CustomLogisticRegressionModel

# Load the saved model
def load_model():
    load_path = "saved_model/logistic_regression_model.pkl"
    model = CustomLogisticRegressionModel.load(load_path)
    return model

model = load_model()

# Get user input
def get_user_input():
    # Collect raw data
    age = st.slider("Age", 18, 70)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    location = st.selectbox('Location', ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston'])
    subscription_length = st.slider("Subscription Length(in months)", 1, 24)
    monthly_bill = st.slider("Monthly Bill($)", 30.00, 100.00)
    total_usage = st.slider("Total Usage(GB)", 50, 500)
    
    # Compute derived features
    if gender == 'Male':
        gender = 1
    else:
        gender = 0
    if location == 'Chicago':
        location = 0
    elif location == 'Houston':
        location = 1
    elif location == 'Los Angeles':
        location = 2
    elif location == 'Miami':
        location = 3
    else:
        location = 4
    avg_usage_per_month = total_usage / subscription_length
    bill_to_usage_ratio = monthly_bill / total_usage
    age_to_usage_ratio = age / total_usage

    # Create the features dictionary
    data = {
        'Age': age,
        'Gender': gender,
        'Location': location,
        'Subscription_Length_Months': subscription_length,
        'Monthly_Bill': monthly_bill,
        'Total_Usage_GB': total_usage,
        'Avg_Usage_per_Month': avg_usage_per_month,
        'Bill_to_Usage_Ratio': bill_to_usage_ratio,
        'Age_to_Usage_Ratio': age_to_usage_ratio
    }
    
    return pd.DataFrame([data])

def main():
    # Creating a sidebar for user input
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox('Page Navigation', ['Home', 'Prediction'])

    st.sidebar.markdown("---")
    st.sidebar.write("Created by [Suryansh Gupta](https://github.com/suryanshgupta9933)")

    if page == 'Home':
        with open('README.md', 'r') as f:
            st.markdown(f.read())

    elif page == 'Prediction':
        st.title('Customer Churn Prediction')
        st.header("Prediction Page")
        user_input = get_user_input()

        # Predict using the model
        if st.button("Predict"):
            prediction = model.predict(user_input)
            probability = model.predict_proba(user_input)

            if prediction[0] == 1:
                st.subheader("Churn: Yes")
                st.write(f"Probability of churn: {probability[0][1]:.2f}")
            else:
                st.subheader("Churn: No")
                st.write(f"Probability of staying: {probability[0][0]:.2f}")

if __name__ == '__main__':
    main()
