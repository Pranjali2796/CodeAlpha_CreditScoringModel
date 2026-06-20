import numpy as np                      # for generating random numbers
import pandas as pd                     # for working with tabular data (DataFrames)
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend so plt.show() doesn't error without a display
import matplotlib.pyplot as plt         # for plotting charts
import seaborn as sns                   # for nicer-looking statistical plots
import joblib                           # for saving/loading the trained model

from sklearn.preprocessing import LabelEncoder        # to convert text -> numbers
from sklearn.model_selection import train_test_split  # to split data
from sklearn.ensemble import RandomForestClassifier    # our ML model
from sklearn.metrics import (
    accuracy_score,        # measures overall correctness
    classification_report, # gives precision, recall, f1-score
    confusion_matrix        # shows correct vs incorrect predictions
)

# Setting a random seed means that every time we run this script,
# we get the SAME "random" data and results. This makes the project
# reproducible (very useful for learning and for showing your work).
np.random.seed(42)



print("Step 1: Generating synthetic dataset...")

n_records = 500  # number of fake "customers" we want to create

# Income: random monthly income between 1,000 and 10,000 (USD)
income = np.random.randint(1000, 10000, n_records)

# Debt: random debt amount between 0 and 5,000 (USD)
debt = np.random.randint(0, 5000, n_records)

# Age: random age between 18 and 65
age = np.random.randint(18, 65, n_records)

# Payment_History: randomly choose one of three categories for each person
payment_history = np.random.choice(
    ["Good", "Average", "Poor"],
    size=n_records,
    p=[0.4, 0.35, 0.25]   # probabilities: 40% Good, 35% Average, 25% Poor
)


credit_status = []

for i in range(n_records):
    score = 0

    # Higher income improves the score
    if income[i] > 5000:
        score += 1

    # Higher debt reduces the score
    if debt[i] > 2500:
        score -= 1

    # Payment history strongly affects the score
    if payment_history[i] == "Good":
        score += 2
    elif payment_history[i] == "Average":
        score += 0
    else:  # "Poor"
        score -= 2

    # Add a small amount of randomness so the data isn't too "perfect"
    score += np.random.choice([-1, 0, 1])

    # If the final score is positive, we say the person has GOOD credit (1)
    # Otherwise, they have BAD credit (0)
    if score > 0:
        credit_status.append(1)
    else:
        credit_status.append(0)

# Combine all the columns into one DataFrame (a table)
df = pd.DataFrame({
    "Income": income,
    "Debt": debt,
    "Age": age,
    "Payment_History": payment_history,
    "Credit_Status": credit_status
})


missing_rows = np.random.choice(df.index, size=10, replace=False)
df.loc[missing_rows[:5], "Income"] = np.nan          # 5 missing Income values
df.loc[missing_rows[5:], "Payment_History"] = np.nan  # 5 missing Payment_History values

# Save the dataset to a CSV file so we have a record of it
df.to_csv("credit_data.csv", index=False)

print(f"Dataset created with {df.shape[0]} rows and {df.shape[1]} columns.")
print("Saved dataset as 'credit_data.csv'")
print("\nFirst 5 rows of the dataset:")
print(df.head())


print("\nStep 2: Preprocessing data...")

# ---------------------------------------------------------
# 2a. Handle missing values
# ---------------------------------------------------------
print(f"\nMissing values BEFORE cleaning:\n{df.isnull().sum()}")

# For the numeric column "Income", fill missing values with the MEDIAN.
# The median is a good choice because it isn't affected much by
# very high or very low outlier values.
df["Income"] = df["Income"].fillna(df["Income"].median())

# For the categorical column "Payment_History", fill missing values
# with the MODE (the most frequently occurring category).
df["Payment_History"] = df["Payment_History"].fillna(df["Payment_History"].mode()[0])

print(f"\nMissing values AFTER cleaning:\n{df.isnull().sum()}")


# Machine learning models only understand numbers, not text.
# LabelEncoder converts text categories into numeric codes.
# Example: "Poor" -> 0, "Average" -> 1, "Good" -> 2 (alphabetical order
# is NOT guaranteed; LabelEncoder assigns numbers based on sorted order)
label_encoder = LabelEncoder()
df["Payment_History"] = label_encoder.fit_transform(df["Payment_History"])

print("\nPayment_History encoding map:")
for category, code in zip(label_encoder.classes_,
                           label_encoder.transform(label_encoder.classes_)):
    print(f"  '{category}' -> {code}")

print("\nDataset after preprocessing:")
print(df.head())



print("\nStep 3: Splitting data into train and test sets...")

# X = the input features (everything the model uses to make a prediction)
# y = the target (what we want to predict: Credit_Status)
X = df.drop("Credit_Status", axis=1)
y = df["Credit_Status"]

# train_test_split randomly divides the data:
#   80% for training (teaching the model)
#   20% for testing (checking how well the model learned)
# random_state=42 makes the split reproducible.
# stratify=y keeps the same proportion of Good/Bad credit in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows")


print("\nStep 4: Training the Random Forest model...")

# Random Forest builds many decision trees and combines their votes
# to make a final prediction. This usually gives better and more
# stable results than a single decision tree.
model = RandomForestClassifier(
    n_estimators=100,    # number of trees in the forest
    random_state=42      # makes results reproducible
)

# .fit() is where the actual "learning" happens
model.fit(X_train, y_train)

print("Model training complete!")



print("\nStep 5: Evaluating the model...")

# Use the trained model to make predictions on the UNSEEN test data
y_pred = model.predict(X_test)


# Accuracy = (number of correct predictions) / (total predictions)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}  ({accuracy * 100:.2f}%)")


# This shows Precision, Recall, and F1-score for each class
# (0 = Bad Credit, 1 = Good Credit), which gives more detail
# than accuracy alone.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"]))


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (raw numbers):")
print(cm)


print("\nStep 6: Creating confusion matrix visualization...")

plt.figure(figsize=(6, 5))

# sns.heatmap() draws the confusion matrix as a colored grid with numbers
sns.heatmap(
    cm,
    annot=True,           # show the numbers on the chart
    fmt="d",              # display numbers as plain integers
    cmap="Blues",         # color scheme
    xticklabels=["Bad Credit", "Good Credit"],
    yticklabels=["Bad Credit", "Good Credit"]
)

plt.title("Confusion Matrix - Random Forest Credit Scoring Model")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

# Save the chart as an image file
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix chart as 'confusion_matrix.png'")

# Display the chart on screen (when running locally with a display)
plt.show()



print("\nStep 7: Saving the trained model...")

# joblib.dump() saves the trained model to a file so we can reuse it
# later WITHOUT having to retrain it from scratch every time.
joblib.dump(model, "credit_model.pkl")

print("Model saved as 'credit_model.pkl'")


print("\n=========================================================")
print(" PROJECT COMPLETE!")
print(" Files created:")
print("   - credit_data.csv       (the dataset)")
print("   - credit_model.pkl      (the trained model)")
print("   - confusion_matrix.png  (evaluation chart)")
print("=========================================================")