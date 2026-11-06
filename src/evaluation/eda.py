# ==========================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 4: EXPLORATORY DATA ANALYSIS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ----------------------------------------------------------
# 1. PROJECT PATHS
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/raw/Customer-Churn-Records.csv"

FIGURE_PATH = PROJECT_ROOT / "reports/figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------
# 3. TARGET DISTRIBUTION
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Exited")

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")

plt.savefig(
    FIGURE_PATH / "01_churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------------------------
# 4. AGE VS CHURN
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Exited",
    multiple="stack",
    bins=30
)

plt.title("Customer Age Distribution by Churn")
plt.savefig(
    FIGURE_PATH / "02_age_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ----------------------------------------------------------
# 5. GEOGRAPHY VS CHURN
# ----------------------------------------------------------

geo_churn = pd.crosstab(
    df["Geography"],
    df["Exited"],
    normalize="index"
) * 100

print("\nGeography-wise Churn Percentage:")
print(geo_churn)

geo_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5)
)

plt.title("Geography-wise Customer Churn")
plt.ylabel("Percentage")
plt.xlabel("Geography")

plt.savefig(
    FIGURE_PATH / "03_geography_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------------------------
# 6. TENURE VS CHURN
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Exited",
    y="Tenure"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Exited")
plt.ylabel("Tenure")

plt.savefig(
    FIGURE_PATH / "04_tenure_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ----------------------------------------------------------
# 7. ACTIVE MEMBERSHIP VS CHURN
# ----------------------------------------------------------

membership_churn = pd.crosstab(
    df["IsActiveMember"],
    df["Exited"],
    normalize="index"
) * 100

print("\nActive Membership vs Churn:")
print(membership_churn)

membership_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Active Membership vs Customer Churn")
plt.ylabel("Percentage")

plt.savefig(
    FIGURE_PATH / "05_membership_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------------------------
# 8. BALANCE VS CHURN
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Exited",
    y="Balance"
)

plt.title("Customer Balance vs Churn")

plt.savefig(
    FIGURE_PATH / "06_balance_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ----------------------------------------------------------
# 9. NUMBER OF PRODUCTS VS CHURN
# ----------------------------------------------------------

product_churn = pd.crosstab(
    df["NumOfProducts"],
    df["Exited"],
    normalize="index"
) * 100

print("\nProducts vs Churn:")
print(product_churn)

product_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Number of Products vs Customer Churn")
plt.ylabel("Percentage")

plt.savefig(
    FIGURE_PATH / "07_products_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ----------------------------------------------------------
# 10. CORRELATION ANALYSIS
# ----------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Customer Churn Correlation Matrix")

plt.savefig(
    FIGURE_PATH / "08_correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nCorrelation with Exited:")

print(
    correlation["Exited"].sort_values(ascending=False)
)


# ----------------------------------------------------------
# 11. COMPLAINT VS CHURN
# ----------------------------------------------------------

complaint_churn = pd.crosstab(
    df["Complain"],
    df["Exited"],
    normalize="index"
) * 100

print("\nComplaint vs Churn:")
print(complaint_churn)

complaint_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Customer Complaints vs Churn")
plt.ylabel("Percentage")

plt.savefig(
    FIGURE_PATH / "09_complaints_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



