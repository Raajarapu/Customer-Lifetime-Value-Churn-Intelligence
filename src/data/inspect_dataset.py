# ============================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 3: DATASET UNDERSTANDING AND QUALITY ASSESSMENT
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Customer-Churn-Records.csv"

REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CUSTOMER CHURN DATA QUALITY ASSESSMENT")
print("=" * 65)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\n1. DATASET INFORMATION")

print("Dataset Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])


# ------------------------------------------------------------
# 3. COLUMN INFORMATION
# ------------------------------------------------------------

print("\n2. COLUMN INFORMATION")

print(df.info())


# ------------------------------------------------------------
# 4. DATA TYPES
# ------------------------------------------------------------

print("\n3. DATA TYPES")

print(df.dtypes)


# ------------------------------------------------------------
# 5. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n4. MISSING VALUE ANALYSIS")

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_report = pd.DataFrame({
    "Missing_Count": missing_values,
    "Missing_Percentage": missing_percentage
})

print(missing_report)


# ------------------------------------------------------------
# 6. DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n5. DUPLICATE ANALYSIS")

duplicate_rows = df.duplicated().sum()

print("Duplicate Rows:", duplicate_rows)

if "CustomerId" in df.columns:
    duplicate_customers = df["CustomerId"].duplicated().sum()

    print("Duplicate Customer IDs:", duplicate_customers)


# ------------------------------------------------------------
# 7. NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

print("\n6. FEATURE CLASSIFICATION")

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numerical Features:", numerical_columns)

print("\nCategorical Features:", categorical_columns)

print("\nNumerical Feature Count:", len(numerical_columns))

print("Categorical Feature Count:", len(categorical_columns))


# ------------------------------------------------------------
# 8. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n7. STATISTICAL SUMMARY")

print(df.describe(include="all").T)


# ------------------------------------------------------------
# 9. TARGET VARIABLE ANALYSIS
# ------------------------------------------------------------

print("\n8. TARGET VARIABLE ANALYSIS")

target = "Exited"

if target in df.columns:

    target_counts = df[target].value_counts()

    target_percentage = df[target].value_counts(normalize=True) * 100

    target_report = pd.DataFrame({
        "Count": target_counts,
        "Percentage": target_percentage
    })

    print(target_report)

else:
    print("Target variable not found!")


# ------------------------------------------------------------
# 10. UNIQUE VALUE ANALYSIS
# ------------------------------------------------------------

print("\n9. UNIQUE VALUE ANALYSIS")

for column in df.columns:

    print(
        f"{column}: {df[column].nunique()} unique values"
    )


# ------------------------------------------------------------
# 11. BUSINESS FEATURE ANALYSIS
# ------------------------------------------------------------

print("\n10. BUSINESS FEATURE ANALYSIS")

business_features = [
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "IsActiveMember",
    "Complain",
    "Exited"
]

for column in business_features:

    if column in df.columns:

        print(f"\n{column}")

        print(df[column].value_counts(dropna=False).head(10))


# ------------------------------------------------------------
# 12. COMPLAINT AND CHURN RELATIONSHIP
# ------------------------------------------------------------

print("\n11. COMPLAINT VS CHURN ANALYSIS")

if "Complain" in df.columns and "Exited" in df.columns:

    complaint_analysis = pd.crosstab(
        df["Complain"],
        df["Exited"],
        normalize="index"
    ) * 100

    print(complaint_analysis)


# ------------------------------------------------------------
# 13. DATA QUALITY SUMMARY
# ------------------------------------------------------------

print("\n12. FINAL DATA QUALITY SUMMARY")

print("Total Records:", len(df))

print("Total Features:", len(df.columns))

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Rows:", duplicate_rows)

print("Dataset Shape:", df.shape)

print("\nDataset assessment completed!")


# ------------------------------------------------------------
# 14. SAVE DATA QUALITY REPORT
# ------------------------------------------------------------

report_file = REPORT_PATH / "data_quality_report.txt"

with open(report_file, "w", encoding="utf-8") as file:

    file.write("CUSTOMER CHURN DATA QUALITY REPORT\n")

    file.write("=" * 50 + "\n")

    file.write(f"\nDataset Shape: {df.shape}\n")

    file.write(f"\nMissing Values:\n{missing_report}\n")

    file.write(f"\nDuplicate Rows: {duplicate_rows}\n")

    file.write(f"\nNumerical Features: {numerical_columns}\n")

    file.write(f"\nCategorical Features: {categorical_columns}\n")

    file.write(f"\nStatistical Summary:\n{df.describe(include='all').T}\n")

    if "Exited" in df.columns:
        file.write(f"\nTarget Distribution:\n{target_report}\n")

print("\nReport saved successfully!")

print(report_file)

print("\n" + "=" * 65)
print("STAGE 3 COMPLETED")
print("=" * 65)