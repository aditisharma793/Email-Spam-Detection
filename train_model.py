import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline

# Step 1: Load cleaned dataset
data = pd.read_csv("dataset/cleaned_sms.csv")

# Step 2: Select input and output
X = data["Cleaned_Message"]
y = data["Category"]

# Step 3: Divide data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 4: Create Naive Bayes pipeline
nb_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Step 5: Train Naive Bayes model
nb_model.fit(X_train, y_train)

# Step 6: Make predictions
nb_predictions = nb_model.predict(X_test)

# Step 7: Evaluate Naive Bayes
print("\nNAIVE BAYES RESULTS")

print("Accuracy:",
      accuracy_score(y_test, nb_predictions))

print("\nClassification Report:")
print(classification_report(y_test, nb_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))

# Step 8: Create Logistic Regression pipeline
lr_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 9: Train Logistic Regression
lr_model.fit(X_train, y_train)

# Step 10: Make predictions
lr_predictions = lr_model.predict(X_test)

# Step 11: Evaluate Logistic Regression
print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy:",
      accuracy_score(y_test, lr_predictions))

print("\nClassification Report:")
print(classification_report(y_test, lr_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

# Step 12: Select the better model
nb_accuracy = accuracy_score(y_test, nb_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

if lr_accuracy > nb_accuracy:
    best_model = lr_model
    best_name = "Logistic Regression"
else:
    best_model = nb_model
    best_name = "Naive Bayes"

# Step 13: Create model folder
os.makedirs("model", exist_ok=True)

# Step 14: Save the selected model
joblib.dump(best_model, "model/spam_model.pkl")

print("\nBest Model:", best_name)
print("Best Accuracy:", max(nb_accuracy, lr_accuracy))
print("\nModel saved successfully!")