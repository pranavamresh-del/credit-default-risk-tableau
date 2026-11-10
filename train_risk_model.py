import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
N_RECORDS = 15000

print("Generating synthetic loan portfolio data...")

# 1. Generate Realistic Financial Features
annual_income = np.random.lognormal(mean=11.0, sigma=0.55, size=N_RECORDS)
annual_income = np.clip(annual_income, 25000, 350000)

loan_amount = np.random.uniform(5000, 45000, size=N_RECORDS)
credit_score = np.random.normal(loc=690, scale=60, size=N_RECORDS).astype(int)
credit_score = np.clip(credit_score, 450, 850)

dti_ratio = (
    (loan_amount / annual_income) * 100
    + np.random.normal(loc=12, scale=5, size=N_RECORDS)
)
dti_ratio = np.clip(dti_ratio, 3.0, 65.0)

revolving_util = np.random.beta(a=2, b=4, size=N_RECORDS) * 100
delinquencies_2yr = np.random.poisson(lam=0.35, size=N_RECORDS)
employment_length = np.random.randint(1, 20, size=N_RECORDS)

# 2. Generate Ground-Truth Default Log-Odds
z = (
    -0.015 * (credit_score - 600)
    + 0.05 * (dti_ratio - 20)
    + 0.025 * (revolving_util - 30)
    + 0.45 * delinquencies_2yr
    - 0.02 * (employment_length - 5)
    - 2.8
)
default_prob = 1 / (1 + np.exp(-z))
loan_status = (np.random.rand(N_RECORDS) < default_prob).astype(int)

df = pd.DataFrame(
    {
        "loan_id": [f"LN-{100000 + i}" for i in range(N_RECORDS)],
        "annual_income": np.round(annual_income, 2),
        "loan_amount": np.round(loan_amount, 2),
        "credit_score": credit_score,
        "dti_ratio": np.round(dti_ratio, 2),
        "revolving_utilization_pct": np.round(revolving_util, 2),
        "delinquencies_2yr": delinquencies_2yr,
        "employment_length_years": employment_length,
        "actual_default": loan_status,
    }
)

# 3. Train Machine Learning Model
feature_cols = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "dti_ratio",
    "revolving_utilization_pct",
    "delinquencies_2yr",
    "employment_length_years",
]

X = df[feature_cols]
y = df["actual_default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=120, max_depth=8, random_state=42, class_weight="balanced"
)
clf.fit(X_train_scaled, y_train)

# 4. Score Entire Dataset for Visualization
X_all_scaled = scaler.transform(X)
predicted_probabilities = clf.predict_proba(X_all_scaled)[:, 1]

df["predicted_default_prob"] = np.round(predicted_probabilities, 4)

# Assign Risk Tiers (Compatible with NumPy 2.0+)
conditions = [
    (df["predicted_default_prob"] < 0.15),
    ((df["predicted_default_prob"] >= 0.15) & (df["predicted_default_prob"] < 0.35)),
    ((df["predicted_default_prob"] >= 0.35) & (df["predicted_default_prob"] < 0.60)),
    (df["predicted_default_prob"] >= 0.60),
]
tiers = ["Low Risk", "Moderate Risk", "High Risk", "Severe Risk"]
df["risk_tier"] = np.select(conditions, tiers, default="Unknown")

# Expected Loss Calculation
# Expected Loss = Probability of Default * Exposure (Loan Amount) * Loss Given Default (45%)
df["expected_loss"] = np.round(
    df["predicted_default_prob"] * df["loan_amount"] * 0.45, 2
)

# 5. Export to CSV for Tableau
output_filename = "credit_risk_tableau_feed.csv"
df.to_csv(output_filename, index=False)

auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])
print(f"Model trained successfully. Test ROC-AUC Score: {auc:.4f}")
print(f"Exported: {output_filename} ready for Tableau connection.")