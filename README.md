# Thief Detector — Anomaly Detection in Financial Transactions

## Overview

This project focuses on detecting suspicious financial transactions using anomaly detection techniques. It is designed for scenarios where labeled fraud data is not available. The system identifies unusual patterns in transaction behavior and flags them as suspicious.

---

## Problem Statement

In real-world financial systems, fraud labels are often unavailable. This makes it difficult to train supervised machine learning models. The goal of this project is to detect anomalous transactions by learning normal behavior and identifying deviations.

---

## Objective

* Detect unusual transactions without labeled data
* Compare different anomaly detection models
* Evaluate model performance using recall

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## Project Workflow

Input Data
→ Data Generation
→ Model Training
→ Anomaly Detection
→ Output (Score + Flag)
→ Visualization

---

## Dataset

Since real transaction data is not accessible due to privacy concerns, synthetic data is generated.

### Features Used

* amount
* hour_of_day
* merchant_category
* user_age
* transactions_last_7_days

### Data Composition

* 1000 normal transactions
* 30 anomalous transactions

---

## Models Used

### 1. Isolation Forest

* Detects anomalies by isolating data points
* Works efficiently for outlier detection

### 2. One-Class SVM

* Learns boundary of normal data
* Flags deviations as anomalies

---

## Evaluation Metric

### Recall

Recall is used to measure how many actual anomalies are correctly detected.

Recall = Detected Anomalies / Total Anomalies

Reason for choosing recall:

* Dataset is imbalanced
* Missing anomalies is critical

---

## Results

* Isolation Forest Recall: 1.0
* One-Class SVM Recall: 0.66

Isolation Forest performed better by detecting all anomalies.

---

## Visualization

A histogram is plotted to show the distribution of anomaly scores:

* Positive scores → Normal transactions
* Negative scores → Anomalies

---

## Function Implementation

### flag_transaction(transaction: dict) -> dict

This function:

* Takes a transaction as input
* Calculates anomaly score
* Returns whether it is suspicious

Example Output:
{
"amount": 12000,
"anomaly_score": -0.12,
"is_suspicious": true
}

---

## How to Run

1. Clone the repository
2. Navigate to the project folder
3. Create virtual environment
   python -m venv venv
4. Activate environment
   venv\Scripts\activate
5. Install dependencies
   pip install numpy pandas scikit-learn matplotlib
6. Run the script
   python thief_detector.py

---

## Project Structure

Thief_Detector/
│
├── thief_detector.py
├── README.md
├── venv/

---

## Limitations

* Uses synthetic data
* Limited number of features
* Not deployed in real-time

---

## Future Scope

* Use real-world transaction data
* Add more features for better accuracy
* Deploy as a real-time fraud detection system
* Integrate with APIs

---

## Conclusion

This project demonstrates how anomaly detection can be used to identify suspicious transactions without labeled data. Among the models tested, Isolation Forest showed better performance.

