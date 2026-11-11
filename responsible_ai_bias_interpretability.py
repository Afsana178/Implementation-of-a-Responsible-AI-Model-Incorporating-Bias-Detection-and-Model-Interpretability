# responsible_ai_bias_interpretability.py
# Implementation of a Responsible AI Model Incorporating Bias Detection and Model Interpretability
# Domain: Artificial Intelligence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# -----------------------------
# 1. Synthetic Dataset Generation (with sensitive attribute)
# -----------------------------
def generate_responsible_ai_data(samples=3000):
    np.random.seed(42)

    age = np.random.randint(18, 65, samples)
    income = np.random.normal(50000, 15000, samples)
    credit_score = np.random.normal(650, 50, samples)

    # Sensitive attribute (e.g., gender encoded as 0/1)
    gender = np.random.choice([0, 1], samples)

    bias_noise = gender * np.random.normal(5000, 2000, samples)

    decision_score = (
        0.3 * income +
        0.4 * credit_score +
        200 * age +
        bias_noise +
        np.random.normal(0, 5000, samples)
    )

    label = (decision_score > np.percentile(decision_score, 60)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "gender": gender,   # sensitive feature
        "target": label
    })

    return df


# -----------------------------
# 2. Dataset Creation & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_responsible_ai_data()
    df.to_csv("responsible_ai_data.csv", index=False)
    print("Synthetic Responsible AI dataset saved.")


# -----------------------------
# 3. Load & Inspect Data
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("responsible_ai_data.csv")
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    return X_scaled, y, X.columns


# -----------------------------
# 5. Model Training
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "responsible_ai_model.pkl")
    print("Model trained and saved")

    return model, X_test, y_test


# -----------------------------
# 6. Bias Detection
# -----------------------------
def bias_analysis(df_test, predictions):
    df_eval = df_test.copy()
    df_eval["prediction"] = predictions

    bias_report = df_eval.groupby("gender")["prediction"].mean()
    print("\nBias Analysis (Positive Prediction Rate by Gender):")
    print(bias_report)

    bias_report.plot(kind="bar", title="Bias Analysis by Gender")
    plt.ylabel("Positive Prediction Rate")
    plt.tight_layout()
    plt.savefig("bias_analysis.png")
    plt.show()


# -----------------------------
# 7. Model Interpretability
# -----------------------------
def interpret_model(model, X_test, y_test, feature_names):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance (Permutation-Based):")
    print(importance_df)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Model Interpretability - Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()


# -----------------------------
# 8. Responsible Inference
# -----------------------------
def responsible_predict(input_features):
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("responsible_ai_model.pkl")

    columns = ["age", "income", "credit_score", "gender"]
    input_df = pd.DataFrame([input_features], columns=columns)
    input_scaled = pd.DataFrame(scaler.transform(input_df),columns=columns)
    prediction = model.predict(input_scaled)

    return prediction[0]

# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    if not os.path.exists("responsible_ai_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X, y, feature_names = preprocess_data(df)

    model, X_test, y_test = train_model(X, y)

    preds = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    bias_analysis(df.loc[y_test.index], preds)
    interpret_model(model, X_test, y_test, feature_names)

    # Pass raw, unscaled values [age, income, credit_score, gender]
    sample = [35, 65000, 720, 1]
    decision = responsible_predict(sample)
    print("\nResponsible AI Prediction Output:", decision)


if __name__ == "__main__":
    main()
