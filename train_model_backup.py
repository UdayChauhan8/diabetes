
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

if __name__ == "__main__":
    df = pd.read_csv('/home/chauhan_817/Downloads/diabetes_1500_patients.csv')

    # =
    cols_with_zero_issue = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero_issue:
        df[col] = df[col].replace(0, df[col].median())

    # ================= FEATURE ENGINEERING (RESTORED) =================
    # CRITICAL: 'app.py' generates these 3 features. If we train without them,
    # the model in app.py will fail or give garbage predictions due to shape mismatch.
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Glucose_Age"] = df["Glucose"] * df["Age"]

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ================= SCALING (RESTORED) =================
    # CRITICAL: 'app.py' loads 'model/scaler.pkl'. We must fit and save it here.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ================= RANDOM FOREST TRAINING =================
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 8, 12, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Using StratifiedKFold for better validation on imbalanced data
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=15,
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best Parameters:", search.best_params_)

    # Evaluation
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # ================= SAVING MODEL =================
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, 'model/rf_model.pkl') # Saved as rf_model.pkl
    joblib.dump(scaler, 'model/scaler.pkl')       # Saved as scaler.pkl

    print("Improved Random Forest model saved to model/rf_model.pkl")