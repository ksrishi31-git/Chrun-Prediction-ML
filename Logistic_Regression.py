import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\user\Downloads\customer_churn_dataset-testing-master.csv")
print("First 5 rows:")
print(df.head())

# 2. Remove Missing Values
df = df.dropna()

# 3. Encode Categorical Columns
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# 4. Separate Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Predict for a New Customer
new_customer = np.array([[
    1001,   # CustomerID
    45,     # Age
    1,      # Gender (Male)
    24,     # Tenure
    15,     # Usage Frequency
    3,      # Support Calls
    5,      # Payment Delay
    2,      # Subscription Type (Standard)
    1,      # Contract Length (Monthly)
    750,    # Total Spend
    10      # Last Interaction
]])

# Scale new customer data
new_customer_scaled = scaler.transform(new_customer)

# Predict
prediction = model.predict(new_customer_scaled)
if prediction[0] == 1:
    print("\nCustomer will CHURN")
else:
    print("\nCustomer will NOT CHURN")
