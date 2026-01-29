import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

if __name__ == "__main__":
    # 1. Load New Dataset
    df = pd.read_csv('/home/chauhan_817/Downloads/diabetes_1500_patients.csv')

    # 2. Define Features
    # Explicitly selecting the 15 features provided by the user + Outcome
    features_list = [
        "FastingGlucose", "HbA1c", "OGTT_2hr", "FastingInsulin", "HOMA_IR",
        "BMI", "WaistCircumference", "WaistHipRatio", "SystolicBP",
        "Triglycerides", "HDL", "Age", "FamilyHistory", "PhysicalActivity", "Sex"
    ]
    
    # Validation: Ensure all columns exist
    for col in features_list:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    X = df[features_list].copy()
    y = df["Outcome"]

    print(f"Features selected: {len(X.columns)}")

    # ================= DATA CLEANING =================
    # Replace 0 with median for biological metrics where 0 is chemically impossible
    # Note: 0 is valid for 'FamilyHistory', 'PhysicalActivity', 'Sex'. 
    cols_to_clean = [
        "FastingGlucose", "HbA1c", "OGTT_2hr", "FastingInsulin", "HOMA_IR",
        "BMI", "WaistCircumference", "WaistHipRatio", "SystolicBP",
        "Triglycerides", "HDL", "Age"
    ]
    
    print("Cleaning data (replacing 0s with median)...")
    for col in cols_to_clean:
        if (X[col] == 0).any():
            median_val = X[col].median()
            X[col] = X[col].replace(0, median_val)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 8, 12, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    print("Running RandomizedSearchCV...")
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc',
        random_state=42,
        n_jobs=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best Parameters:", search.best_params_)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save Feature Names for App Validation
    os.makedirs("model", exist_ok=True)
    joblib.dump(features_list, 'model/feature_names.pkl')
    
    # Save Model & Scaler
    joblib.dump(best_model, 'model/rf_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

    print("Model saved to model/rf_model.pkl")