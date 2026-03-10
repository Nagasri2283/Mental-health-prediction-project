import joblib
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from preprocess import load_data

# -------------------------------
# Load Dataset
# -------------------------------

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "mental_health_prediction.csv")

X_train, X_test, Y_train, Y_test = load_data(data_path)

# -------------------------------
# Compare Multiple Models
# -------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, Y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(Y_test, pred)

    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Model Accuracy:", best_accuracy)

# -------------------------------
# Cross Validation
# -------------------------------

cv_scores = cross_val_score(best_model, X_train, Y_train, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

# -------------------------------
# Feature Importance (if RandomForest)
# -------------------------------

if isinstance(best_model, RandomForestClassifier):

    features = X_train.columns
    importance = best_model.feature_importances_

    plt.figure(figsize=(8,5))
    plt.barh(features, importance)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

# -------------------------------
# Save Model
# -------------------------------

model_path = os.path.join(current_dir, "..", "model", "depression_model.pkl")

joblib.dump(best_model, model_path)

print("\nModel saved successfully at:", model_path)
