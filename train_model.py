"""
Loan Prediction Model Training Script
Generates synthetic loan data, trains a Random Forest classifier,
and saves the model + encoders for the Streamlit app.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. Synthetic Dataset Generation
# ─────────────────────────────────────────────
def generate_loan_dataset(n=5000):
    """Generate a realistic synthetic loan dataset."""
    data = {
        "Gender":             np.random.choice(["Male", "Female"], n, p=[0.65, 0.35]),
        "Married":            np.random.choice(["Yes", "No"],       n, p=[0.65, 0.35]),
        "Dependents":         np.random.choice(["0", "1", "2", "3+"], n, p=[0.45, 0.22, 0.18, 0.15]),
        "Education":          np.random.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
        "Self_Employed":      np.random.choice(["Yes", "No"],       n, p=[0.15, 0.85]),
        "ApplicantIncome":    np.random.lognormal(mean=8.5,  sigma=0.7,  size=n).astype(int),
        "CoapplicantIncome":  np.random.lognormal(mean=7.5,  sigma=1.0,  size=n).astype(int) * np.random.choice([0, 1], n, p=[0.35, 0.65]),
        "LoanAmount":         np.random.lognormal(mean=5.0,  sigma=0.6,  size=n).astype(int),
        "Loan_Amount_Term":   np.random.choice([120, 180, 240, 300, 360, 480], n, p=[0.04, 0.08, 0.05, 0.06, 0.70, 0.07]),
        "Credit_History":     np.random.choice([1.0, 0.0], n, p=[0.84, 0.16]),
        "Property_Area":      np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.33, 0.39, 0.28]),
    }

    df = pd.DataFrame(data)

    # Deterministic target with realistic rules
    score = (
        (df["Credit_History"] == 1.0).astype(int) * 4 +
        (df["Education"] == "Graduate").astype(int) * 2 +
        (df["Married"] == "Yes").astype(int) * 1 +
        (df["Self_Employed"] == "No").astype(int) * 1 +
        (df["Property_Area"] == "Semiurban").astype(int) * 1 +
        (df["ApplicantIncome"] > 5000).astype(int) * 1 +
        (df["LoanAmount"] < 150).astype(int) * 1
    )
    noise = np.random.normal(0, 0.5, n)
    prob  = 1 / (1 + np.exp(-(score - 5 + noise)))
    df["Loan_Status"] = np.where(prob > 0.5, "Y", "N")

    return df

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
def preprocess(df):
    """Encode categoricals and scale numerics."""
    df = df.copy()
    cat_cols = ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area"]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_enc = LabelEncoder()
    df["Loan_Status"] = target_enc.fit_transform(df["Loan_Status"])
    encoders["Loan_Status"] = target_enc

    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, encoders, scaler

# ─────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Loan Prediction – Random Forest Training")
    print("=" * 60)

    # Dataset
    df_raw = generate_loan_dataset(5000)
    print(f"\n✅  Dataset generated: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
    print(f"    Approval rate: {(df_raw['Loan_Status']=='Y').mean()*100:.1f}%")

    df, encoders, scaler = preprocess(df_raw)

    feature_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]
    X = df[feature_cols].values
    y = df["Loan_Status"].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE for class balance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"\n✅  SMOTE applied: {len(X_train_res)} training samples")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_res, y_train_res)

    # Evaluation
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    cv     = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

    print(f"\n📊  Test Accuracy : {acc*100:.2f}%")
    print(f"📊  ROC-AUC Score : {auc:.4f}")
    print(f"📊  CV Accuracy   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Rejected','Approved'])}")

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n🌲  Feature Importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"   {feat:<22} {imp:.4f}  {bar}")

    # ── Save artefacts ──
    os.makedirs("model", exist_ok=True)
    joblib.dump(rf,       "model/loan_rf_model.pkl")
    joblib.dump(encoders, "model/encoders.pkl")
    joblib.dump(scaler,   "model/scaler.pkl")

    # Save dataset for EDA tab
    df_raw.to_csv("model/loan_data.csv", index=False)

    # Save metrics for dashboard
    metrics = {
        "accuracy": round(acc * 100, 2),
        "auc":      round(auc, 4),
        "cv_mean":  round(cv.mean() * 100, 2),
        "cv_std":   round(cv.std() * 100, 2),
        "feature_names":        feature_cols,
        "feature_importances":  rf.feature_importances_.tolist(),
        "confusion_matrix":     confusion_matrix(y_test, y_pred).tolist(),
        "n_samples":            len(df_raw),
        "approval_rate":        round((df_raw["Loan_Status"] == "Y").mean() * 100, 1),
    }
    joblib.dump(metrics, "model/metrics.pkl")

    print("\n✅  Saved: model/loan_rf_model.pkl")
    print("✅  Saved: model/encoders.pkl")
    print("✅  Saved: model/scaler.pkl")
    print("✅  Saved: model/loan_data.csv")
    print("✅  Saved: model/metrics.pkl")
    print("\n🚀  Run:  streamlit run app.py")

if __name__ == "__main__":
    train()
