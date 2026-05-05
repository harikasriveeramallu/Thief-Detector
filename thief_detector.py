import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Generate Data
# -------------------------------
np.random.seed(42)

normal_data = pd.DataFrame({
    "amount": np.random.normal(2000, 500, 1000),
    "hour_of_day": np.random.randint(6, 22, 1000),
    "merchant_category": np.random.randint(1, 10, 1000),
    "user_age": np.random.randint(20, 60, 1000),
    "transactions_last_7_days": np.random.randint(5, 50, 1000)
})

anomaly_data = pd.DataFrame({
    "amount": np.random.normal(10000, 2000, 30),
    "hour_of_day": np.random.choice([0,1,2,3,4,23], 30),
    "merchant_category": np.random.randint(10, 15, 30),
    "user_age": np.random.randint(18, 25, 30),
    "transactions_last_7_days": np.random.randint(0, 5, 30)
})

data = pd.concat([normal_data, anomaly_data], ignore_index=True)
labels = np.array([0]*1000 + [1]*30)

# -------------------------------
# 2. Train Models
# -------------------------------
iso_model = IsolationForest(contamination=0.03, random_state=42)
iso_preds = iso_model.fit_predict(data)
iso_preds = np.where(iso_preds == -1, 1, 0)

svm_model = OneClassSVM(nu=0.03)
svm_preds = svm_model.fit_predict(data)
svm_preds = np.where(svm_preds == -1, 1, 0)

# -------------------------------
# 3. Evaluation
# -------------------------------
iso_recall = recall_score(labels, iso_preds)
svm_recall = recall_score(labels, svm_preds)

print("Isolation Forest Recall:", iso_recall)
print("One-Class SVM Recall:", svm_recall)

# -------------------------------
# 4. Plot Graph
# -------------------------------
scores = iso_model.decision_function(data)

plt.hist(scores, bins=50)
plt.title("Anomaly Score Distribution")
plt.show()

# -------------------------------
# 5. Function
# -------------------------------
def flag_transaction(transaction: dict) -> dict:
    features = np.array([[
        transaction["amount"],
        transaction["hour_of_day"],
        transaction["merchant_category"],
        transaction["user_age"],
        transaction["transactions_last_7_days"]
    ]])

    score = iso_model.decision_function(features)[0]
    pred = iso_model.predict(features)[0]

    transaction["anomaly_score"] = float(score)
    transaction["is_suspicious"] = True if pred == -1 else False

    return transaction

# -------------------------------
# 6. Test Cases
# -------------------------------
test_cases = [
    {"amount": 1500, "hour_of_day": 14, "merchant_category": 3, "user_age": 30, "transactions_last_7_days": 20},
    {"amount": 12000, "hour_of_day": 2, "merchant_category": 12, "user_age": 22, "transactions_last_7_days": 2},
    {"amount": 1800, "hour_of_day": 10, "merchant_category": 5, "user_age": 40, "transactions_last_7_days": 25},
    {"amount": 9000, "hour_of_day": 1, "merchant_category": 11, "user_age": 19, "transactions_last_7_days": 1},
    {"amount": 2200, "hour_of_day": 16, "merchant_category": 4, "user_age": 35, "transactions_last_7_days": 30}
]

for t in test_cases:
    print(flag_transaction(t))

# -------------------------------
# 7. Notes
# -------------------------------
# Before production:
# 1. Need real fraud dataset
# 2. Proper preprocessing pipeline
# 3. Continuous retraining
# 4. Handle false positives