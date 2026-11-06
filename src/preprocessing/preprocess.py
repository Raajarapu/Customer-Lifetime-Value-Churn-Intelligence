# ============================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 5: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/raw/Customer-Churn-Records.csv"

PROCESSED_PATH = PROJECT_ROOT / "data/processed"

MODEL_PATH = PROJECT_ROOT / "models"

REPORT_PATH = PROJECT_ROOT / "reports/evaluation"

for path in [PROCESSED_PATH, MODEL_PATH, REPORT_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("STAGE 5: DATA PREPROCESSING")
print("=" * 65)

print("\nOriginal Dataset Shape:", df.shape)


# ============================================================
# 3. REMOVE IRRELEVANT FEATURES
# ============================================================

remove_columns = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "Complain"
]

existing_remove_columns = [
    col for col in remove_columns if col in df.columns
]

customer_ids = (
    df["CustomerId"].copy()
    if "CustomerId" in df.columns
    else pd.Series(df.index, name="CustomerId")
)

df = df.drop(columns=existing_remove_columns)

print("\nRemoved Columns:", existing_remove_columns)


# ============================================================
# 4. SEPARATE TARGET VARIABLE
# ============================================================

target = "Exited"

if target not in df.columns:
    raise ValueError("Target variable 'Exited' not found!")

X = df.drop(columns=[target]).copy()

y = df[target].copy()

if y.isna().any():
    raise ValueError("Target variable contains missing values.")

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Unexpected target values detected.")

print("\nFeature Shape:", X.shape)

print("Target Shape:", y.shape)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# Age Group

X["AgeGroup"] = pd.cut(
    X["Age"],
    bins=[0, 25, 35, 45, 55, 65, np.inf],
    labels=[
        "Young",
        "Adult",
        "Middle_Aged",
        "Senior",
        "Older",
        "Elderly"
    ],
    right=True
)

# Tenure Group

X["TenureGroup"] = pd.cut(
    X["Tenure"],
    bins=[-1, 2, 5, 8, 10],
    labels=[
        "New_Customer",
        "Growing_Customer",
        "Established_Customer",
        "Long_Term_Customer"
    ]
)

# Zero Balance Indicator

X["ZeroBalance"] = (
    X["Balance"] == 0
).astype(int)

print("\nFeature Engineering Completed")

print(
    "New Features:",
    ["AgeGroup", "TenureGroup", "ZeroBalance"]
)


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed")

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))


# ============================================================
# 8. PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 9. FIT ON TRAINING DATA ONLY
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

print("\nPreprocessing Completed")


# ============================================================
# 10. SAVE PROCESSED DATA
# ============================================================

X_train_processed.to_csv(
    PROCESSED_PATH / "X_train.csv",
    index=False
)

X_test_processed.to_csv(
    PROCESSED_PATH / "X_test.csv",
    index=False
)

y_train.to_csv(
    PROCESSED_PATH / "y_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_PATH / "y_test.csv",
    index=False
)

id_train.to_csv(
    PROCESSED_PATH / "customer_id_train.csv",
    index=False
)

id_test.to_csv(
    PROCESSED_PATH / "customer_id_test.csv",
    index=False
)


# ============================================================
# 11. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    MODEL_PATH / "preprocessor.pkl"
)


# ============================================================
# 12. EVALUATION
# ============================================================

print("\n" + "=" * 65)

print("STAGE 5 EVALUATION")

print("=" * 65)

print("\nTraining Dataset:", X_train_processed.shape)

print("Testing Dataset:", X_test_processed.shape)

print(
    "\nTraining Churn Distribution:"
)

print(y_train.value_counts(normalize=True) * 100)

print(
    "\nTesting Churn Distribution:"
)

print(y_test.value_counts(normalize=True) * 100)

print(
    "\nMissing Values in Training:",
    X_train_processed.isnull().sum().sum()
)

print(
    "Missing Values in Testing:",
    X_test_processed.isnull().sum().sum()
)

print(
    "\nTrain-Test Feature Consistency:",
    list(X_train_processed.columns)
    == list(X_test_processed.columns)
)


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\nAll processed datasets saved successfully!")

print("\nStage 5 Completed Successfully!")

print("=" * 65)