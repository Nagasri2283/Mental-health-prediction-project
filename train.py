"""
train.py — Generates a synthetic mental-health-risk dataset (including age and
gender, matching the original project's form fields), trains a
RandomForestClassifier, saves the model to model.pkl, and saves a
feature-importance bar chart to static/feature_importance.png.

NOTE ON THE DATA: there was no existing labeled dataset, so this uses a
synthetically generated one. The relationships between features and risk are
based on commonly cited factors (sleep, stress, social support, activity,
work hours) so the model behaves sensibly for a demo, but this is a
portfolio project, not a diagnostic tool. To use a real dataset, replace
generate_dataset() with pandas.read_csv() using the same column names.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

RANDOM_SEED = 42
N_SAMPLES = 3000

FEATURE_COLUMNS = [
    "age",
    "gender",              # 0 = Male, 1 = Female (matches the original form)
    "sleep_hours",
    "stress_level",        # 1-10
    "social_support",      # 1-10
    "physical_activity",   # 1-10
    "work_hours",
]

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "sleep_hours": "Sleep Hours",
    "stress_level": "Stress Level",
    "social_support": "Social Support",
    "physical_activity": "Physical Activity",
    "work_hours": "Work Hours",
}


def generate_dataset(n=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(32, 9, n), 18, 65)
    gender = rng.integers(0, 2, n)  # 0 or 1
    sleep_hours = np.clip(rng.normal(6.8, 1.4, n), 3, 10)
    stress_level = np.clip(rng.normal(5.5, 2.2, n), 1, 10)
    social_support = np.clip(rng.normal(6, 2.3, n), 1, 10)
    physical_activity = np.clip(rng.normal(5, 2.3, n), 1, 10)
    work_hours = np.clip(rng.normal(45, 12, n), 20, 80)

    # Weighted risk score: less sleep, more stress, less social support,
    # less activity, and more work hours all push risk up. Age/gender have
    # only mild, mostly-noise-level effects (kept intentionally weak, since
    # in reality these shouldn't dominate the prediction).
    risk_score = (
        (7 - sleep_hours) * 1.3
        + stress_level * 1.6
        - social_support * 1.1
        - physical_activity * 0.8
        + (work_hours - 40) * 0.05
        + (age - 32) * -0.02
        + rng.normal(0, 2.5, n)  # noise
    )

    low_cut, high_cut = np.percentile(risk_score, [40, 75])
    labels = np.where(risk_score <= low_cut, "Low",
              np.where(risk_score <= high_cut, "Moderate", "High"))

    df = pd.DataFrame({
        "age": age.round(0),
        "gender": gender,
        "sleep_hours": sleep_hours.round(1),
        "stress_level": stress_level.round(1),
        "social_support": social_support.round(1),
        "physical_activity": physical_activity.round(1),
        "work_hours": work_hours.round(1),
        "risk_label": labels,
    })
    return df


def save_feature_importance_chart(model, feature_columns, path):
    importances = model.feature_importances_
    labels = [FEATURE_LABELS[c] for c in feature_columns]
    order = np.argsort(importances)

    plt.figure(figsize=(7, 4.5))
    plt.barh([labels[i] for i in order], [importances[i] for i in order], color="#667eea")
    plt.xlabel("Relative importance")
    plt.title("What drives the prediction")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train_and_save():
    df = generate_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": FEATURE_COLUMNS},
        "model.pkl",
    )
    print("Saved model.pkl")

    save_feature_importance_chart(model, FEATURE_COLUMNS, "static/feature_importance.png")
    print("Saved static/feature_importance.png")


if __name__ == "__main__":
    train_and_save()
