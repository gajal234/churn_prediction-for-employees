import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("C:\\Users\\acer\\Downloads\\customer_churn_data.csv")

# Create a dict to store encoders for each categorical column
label_encoders = {}
categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "Contract", "Churn"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # store the encoder

# Split into features and target
X = df.drop(["Churn"], axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)

# Save model and preprocessing objects (including column order)
X_columns = X.columns.tolist()  # store the training column order

pickle.dump({
    "model": lr,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "X_columns": X_columns
}, open('churn_prediction.pkl', 'wb'))



# Prediction function
def predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges):
    # Create DataFrame for the new input
    data = {
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "Contract": [Contract],
        "TotalCharges": [TotalCharges]
    }
    df1 = pd.DataFrame(data)

    # Encode categorical columns using the saved encoders
    for col in ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "Contract"]:
        df1[col] = label_encoders[col].transform(df1[col])

    # ✅ Step 2: Convert SeniorCitizen to numeric if needed
    if df1["SeniorCitizen"].dtype == object:
        df1["SeniorCitizen"] = df1["SeniorCitizen"].map({"Yes": 1, "No": 0})

    # ✅ Step 3: Match column order to training data
    df1 = df1[X_columns]  # X_columns will come from the pickle in actual deployment

    # Scale features
    df1_scaled = scaler.transform(df1)

    # Predict churn
    result = lr.predict(df1_scaled)
    return result[0]


# Example usage
result = predict_churn("Female", 0, "Yes", "No", 1, "No", "No phone service", "Month-to-month", 29.85)
if result == 0:
    print("not churn")
else:
    print("churn")
