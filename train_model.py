import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

data = {
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Male'],
    'Married': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Dependents': ['0', '1', '0', '0', '0', '2', '0', '3+', '1', '1'],
    'Education': ['Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Graduate'],
    'Self_Employed': ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No'],
    'ApplicantIncome': [5849, 4583, 3000, 2583, 6000, 5417, 2333, 3036, 4006, 12841],
    'CoapplicantIncome': [0, 1508, 0, 2358, 0, 4196, 1516, 2504, 1526, 10968],
    'LoanAmount': [120, 128, 66, 120, 141, 267, 95, 158, 168, 349],
    'Loan_Amount_Term': [360, 360, 360, 360, 360, 360, 360, 360, 360, 360],
    'Credit_History': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    'Property_Area': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban', 'Urban', 'Urban', 'Semiurban', 'Urban', 'Semiurban'],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N']
}

# Try to load existing csv, otherwise use dummy
try:
    df = pd.read_csv('loan_data.csv')
    print("Used external CSV file.")
except:
    df = pd.DataFrame(data)
    print("Used dummy data for training.")

# 2. Preprocessing
# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Encode Categorical Data manually to keep track of mapping
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Fix Dependents (handle "3+")
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# 3. Train Model
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Save the Model
joblib.dump(model, 'loan_model.pkl')
print("Model trained and saved as 'loan_model.pkl'")