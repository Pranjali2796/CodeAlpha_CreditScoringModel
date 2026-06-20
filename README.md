# Credit Scoring Model

## Overview

This project is developed as part of the CodeAlpha Machine Learning Internship.

The Credit Scoring Model predicts whether a customer is creditworthy based on financial information such as income, debt, age, and payment history. The model uses Machine Learning techniques to classify customers into good or bad credit categories.

---

## Objective

To predict an individual's creditworthiness using historical financial data and machine learning algorithms.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

---

## Dataset Features

* Income
* Debt
* Age
* Payment History
* Credit Status (Target Variable)

---

## Machine Learning Model

* Random Forest Classifier

---

## Project Workflow

1. Load and preprocess the dataset.
2. Handle missing values.
3. Encode categorical features.
4. Split data into training and testing sets.
5. Train the Random Forest model.
6. Evaluate model performance.
7. Generate confusion matrix.
8. Save the trained model.

---

## Evaluation Metrics

* Accuracy Score
* Classification Report
* Confusion Matrix

---

## Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── credit_scoring.py
├── credit_data.csv
├── credit_model.pkl
├── confusion_matrix.png
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python credit_scoring.py
```

---

## Output

* Trained Credit Scoring Model
* Accuracy Score
* Classification Report
* Confusion Matrix Visualization

---

