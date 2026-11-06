import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from causalml.inference.meta import BaseTClassifier
from causalml.metrics import get_qini


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/external/hillstrom.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset:", df.shape)


# --------------------------------------------------
# 2. SELECT TREATMENT AND CONTROL GROUPS
# --------------------------------------------------

df = df[df["segment"].isin(
    ["Womens E-Mail", "No E-Mail"]
)].copy()

df["treatment"] = (
    df["segment"] == "Womens E-Mail"
).astype(int)

print("\nSelected dataset:", df.shape)

print("\nTreatment distribution:")
print(df["treatment"].value_counts())


# --------------------------------------------------
# 3. DEFINE FEATURES AND OUTCOME
# --------------------------------------------------

# Only pre-campaign customer information.
# Exclude post-campaign outcomes and campaign labels.

drop_columns = [
    "segment",
    "visit",
    "conversion",
    "spend",
    "treatment"
]

X = df.drop(columns=drop_columns)
T = df["treatment"]
Y = df["conversion"]

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 4. PREPROCESS FEATURES
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Split before fitting preprocessing to avoid leakage.

X_train_raw, X_test_raw, T_train, T_test, Y_train, Y_test = (
    train_test_split(
        X,
        T,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=T
    )
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)


# --------------------------------------------------
# 5. BUILD T-LEARNER
# --------------------------------------------------

base_learner = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)

uplift_model = BaseTClassifier(
    learner=base_learner,
    control_name=0
)

print("\nTraining T-Learner...")

uplift_model.fit(
    X_train,
    treatment=T_train.to_numpy(),
    y=Y_train.to_numpy()
)

print("Model training completed!")


# --------------------------------------------------
# 6. PREDICT CUSTOMER UPLIFT
# --------------------------------------------------

uplift_predictions = uplift_model.predict(X_test)

uplift_predictions = np.asarray(
    uplift_predictions
).reshape(-1)

results = pd.DataFrame({
    "treatment": T_test.to_numpy(),
    "conversion": Y_test.to_numpy(),
    "uplift": uplift_predictions
})

results["uplift_category"] = pd.qcut(
    results["uplift"].rank(method="first"),
    q=4,
    labels=[
        "Low Uplift",
        "Medium Uplift",
        "High Uplift",
        "Very High Uplift"
    ]
)

print("\nUplift predictions:")
print(results.head())


# --------------------------------------------------
# 7. QINI EVALUATION
# --------------------------------------------------

qini_curve = get_qini(
    results,
    outcome_col="conversion",
    treatment_col="treatment",
    treatment_effect_col="uplift"
)

qini_curve.to_csv(
    "reports/evaluation/qini_curve.csv"
)

print("\nQini curve:")
print(qini_curve.tail())


# --------------------------------------------------
# 8. DECILE / SEGMENT ANALYSIS
# --------------------------------------------------

segment_summary = results.groupby(
    "uplift_category",
    observed=False
).agg(
    customers=("conversion", "count"),
    treatment_customers=("treatment", "sum"),
    conversions=("conversion", "sum"),
    average_uplift=("uplift", "mean")
).reset_index()

segment_summary.to_csv(
    "reports/evaluation/uplift_segment_summary.csv",
    index=False
)

print("\nUplift segment summary:")
print(segment_summary)


# --------------------------------------------------
# 9. VISUALIZE QINI CURVE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for column in qini_curve.columns:
    plt.plot(
        qini_curve.index,
        qini_curve[column],
        label=column
    )

plt.title("Uplift Model - Qini Curve")
plt.xlabel("Customer Population")
plt.ylabel("Cumulative Incremental Gain")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/figures/20_uplift_qini_curve.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. SAVE MODEL AND PREPROCESSOR
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("reports/evaluation", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

joblib.dump(
    uplift_model,
    "models/uplift_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/uplift_preprocessor.pkl"
)

results.to_csv(
    "data/processed/uplift_predictions.csv",
    index=False
)

print("\n--------------------------------")
print("UPLIFT MODELLING COMPLETED")
print("--------------------------------")
print("Model saved.")
print("Preprocessor saved.")
print("Qini curve saved.")
print("Customer uplift predictions saved.")
