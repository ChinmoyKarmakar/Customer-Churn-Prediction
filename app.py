# Importing Dependencies
import streamlit as st

# Defining the app
st.title("Customer Churn Prediction")

# Creating a sidebar for user input
st.sidebar.header("User Input Features")
page = st.sidebar.selectbox('Page Navigation', ['Home', 'Prediction'])

st.sidebar.markdown("---")
st.sidebar.write("Created by [Suryansh Gupta](https://github.com/suryanshgupta9933)")

if page == 'Home':
    st.write("Home Page")
if page == 'Prediction':
    st.header("Prediction Page")
    age = st.slider("Age", 18, 70)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    location = st.selectbox('Location', ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston'])
    tenure = st.slider("Subscription Length(in months)", 1, 24)
    bill = st.slider("Monthly Bill($)", 30.00, 100.00)
    usage = st.slider("Monthly Usage(GB)", 50, 500)

    if st.button("Predict"):
        st.write("Prediction")
        st.write("Churn Probability: 0.2")
        st.write("Churn: No")
        st.write("Retention Strategy: Offer Discount")