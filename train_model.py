import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')

    # ================= DATA CLEANING (MAJOR FIX) =================
    cols_with_zero_issue = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero_issue:
        df[col] = df[col].replace(0, df[col].median())

    # ================= FEATURE ENGINEERING =================
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Glucose_Age"] = df["Glucose"] * df["Age"]

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(probability=True, class_weight='balanced')

    param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=cv,
        scoring='roc_auc',  # better than recall alone
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, 'model/svm_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

    print("Improved model saved.")
