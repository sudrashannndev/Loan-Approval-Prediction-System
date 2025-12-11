import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
try:
    model = joblib.load('loan_model.pkl')
except:
    st.error("Model not found! Please run 'train_model.py' first.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Approval AI", layout="wide")

# --- HEADER ---
st.title("🏦 Loan Approval Prediction System")
st.markdown("### Introduction to Data Science Project (CSL-487)")
st.write("This AI system predicts whether a loan should be **Approved** or **Rejected** based on applicant details.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("📝 Applicant Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    married = st.sidebar.selectbox("Married", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
    
    applicant_income = st.sidebar.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.sidebar.number_input("Loan Amount (Thousands)", min_value=0, value=120)
    loan_term = st.sidebar.selectbox("Loan Term (Days)", (360, 180, 120, 60))
    credit_history = st.sidebar.selectbox("Credit History", ("Good (1.0)", "Bad (0.0)"))
    property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    # Convert inputs to match model training format
    data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 1 if education == "Graduate" else 0,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1.0 if "Good" in credit_history else 0.0,
        'Property_Area': 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get User Input
input_df = user_input_features()

# --- MAIN SECTION TABS ---
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Data Analysis"])

with tab1:
    st.subheader("Applicant Summary")
    st.write(input_df)

    if st.button("Predict Loan Status"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        if prediction[0] == 1:
            st.success("✅ CONGRATULATIONS! Loan Status: APPROVED")
            st.balloons()
        else:
            st.error("❌ SORRY! Loan Status: REJECTED")
        
        st.write(f"Confidence Score: {np.max(probability)*100:.2f}%")

with tab2:
    st.subheader("Dataset Analysis (Exploratory Data Analysis)")
    st.info("This section visualizes the training data used for this project.")
    
    # Try to load the original CSV for visualization
    try:
        # If you have the real csv, ensure it is named 'loan_data.csv'
        raw_data = pd.read_csv('LoanApprovalPrediction.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Loan Approval Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='Loan_Status', data=raw_data, ax=ax1, palette='viridis')
            st.pyplot(fig1)

        with col2:
            st.write("#### Income vs Loan Status")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=raw_data, ax=ax2)
            st.pyplot(fig2)
            
        st.write("#### Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10,5))
        numeric_df = raw_data.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

    except:
        st.warning("Please upload 'loan_data.csv' to the project folder to see the visualizations.")

# --- FOOTER ---
st.markdown("---")
st.markdown("Project by **Sudrashan** | Reg No: 02-134231-106")