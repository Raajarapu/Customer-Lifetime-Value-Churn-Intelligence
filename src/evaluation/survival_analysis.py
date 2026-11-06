# ============================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 8: SURVIVAL DATA FEASIBILITY ASSESSMENT
# ============================================================

import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/Customer-Churn-Records.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("STAGE 8: SURVIVAL ANALYSIS DATA ASSESSMENT")
print("=" * 65)

# 1. Dataset information

print("\nDataset Shape:", df.shape)

# 2. Survival-related columns

survival_columns = [
    "Tenure",
    "Exited"
]

for column in survival_columns:

    if column in df.columns:

        print(f"\n{column} Information")

        print(df[column].describe())

        print("\nUnique Values:")

        print(df[column].value_counts().head(15))


# 3. Check available date columns

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
    or "time" in col.lower()
]

print("\nDate/Time Columns:", date_columns)


# 4. Check event distribution

print("\nCustomer Event Distribution")

print(df["Exited"].value_counts())

print("\nChurn Percentage")

print(df["Exited"].mean() * 100)


# 5. Check tenure distribution by churn

print("\nTenure by Customer Status")

print(
    df.groupby("Exited")["Tenure"].describe()
)


# 6. Final assessment

print("\n" + "=" * 65)

print("SURVIVAL DATA FEASIBILITY ASSESSMENT COMPLETED")

print("=" * 65)
