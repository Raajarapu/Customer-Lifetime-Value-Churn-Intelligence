# ============================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 9: CUSTOMER VALUE ESTIMATION AND SEGMENTATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/Customer-Churn-Records.csv"

OUTPUT_PATH = ROOT / "data/processed"

MODEL_PATH = ROOT / "models"

REPORT_PATH = ROOT / "reports/evaluation"

FIGURE_PATH = ROOT / "reports/figures"

for path in [
    OUTPUT_PATH,
    MODEL_PATH,
    REPORT_PATH,
    FIGURE_PATH
]:
    path.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("STAGE 9: CUSTOMER VALUE AND SEGMENTATION")
print("=" * 65)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded:", df.shape)


# ============================================================
# 3. CUSTOMER VALUE PROXY
# ============================================================

# This is an illustrative business proxy.
# It is NOT actual revenue or profit-based CLV.

df["Balance_Value"] = np.log1p(
    df["Balance"].clip(lower=0)
)

df["Salary_Value"] = np.log1p(
    df["EstimatedSalary"].clip(lower=0)
)

df["Product_Value"] = (
    df["NumOfProducts"] / df["NumOfProducts"].max()
)

df["Activity_Value"] = df["IsActiveMember"]

df["Customer_Value_Score"] = (

    0.40 * df["Balance_Value"].rank(pct=True)

    + 0.20 * df["Salary_Value"].rank(pct=True)

    + 0.25 * df["Product_Value"]

    + 0.15 * df["Activity_Value"]

)


# ============================================================
# 4. CUSTOMER VALUE SEGMENTATION
# ============================================================

df["Value_Segment"] = pd.qcut(

    df["Customer_Value_Score"].rank(method="first"),

    q=4,

    labels=[
        "Low Value",
        "Medium Value",
        "High Value",
        "Premium Value"
    ]

)


print("\nCustomer Value Segments:")

print(
    df["Value_Segment"].value_counts()
)


# ============================================================
# 5. PREPARE CLUSTERING FEATURES
# ============================================================

clustering_features = [

    "Age",

    "Tenure",

    "Balance",

    "EstimatedSalary",

    "NumOfProducts",

    "IsActiveMember",

    "CreditScore"

]

cluster_data = df[clustering_features].copy()


# ============================================================
# 6. FEATURE TRANSFORMATION
# ============================================================

cluster_data["Balance"] = np.log1p(
    cluster_data["Balance"].clip(lower=0)
)

cluster_data["EstimatedSalary"] = np.log1p(
    cluster_data["EstimatedSalary"].clip(lower=0)
)


# ============================================================
# 7. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    cluster_data
)


# ============================================================
# 8. FIND OPTIMAL CLUSTERS
# ============================================================

silhouette_results = []

for k in range(2, 7):

    kmeans = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    labels = kmeans.fit_predict(
        scaled_data
    )

    score = silhouette_score(
        scaled_data,
        labels
    )

    silhouette_results.append({

        "Clusters": k,

        "Silhouette_Score": score

    })

    print(
        f"K={k} | Silhouette Score={score:.4f}"
    )


silhouette_df = pd.DataFrame(
    silhouette_results
)

silhouette_df.to_csv(

    REPORT_PATH / "clustering_evaluation.csv",

    index=False

)


# ============================================================
# 9. TRAIN FINAL K-MEANS MODEL
# ============================================================

best_k = int(

    silhouette_df.loc[
        silhouette_df["Silhouette_Score"].idxmax(),
        "Clusters"
    ]

)

print("\nSelected Clusters:", best_k)


final_model = KMeans(

    n_clusters=best_k,

    random_state=42,

    n_init=10

)

df["Customer_Cluster"] = final_model.fit_predict(
    scaled_data
)


# ============================================================
# 10. SAVE CLUSTERING MODELS
# ============================================================

joblib.dump(

    scaler,

    MODEL_PATH / "customer_scaler.pkl"

)

joblib.dump(

    final_model,

    MODEL_PATH / "customer_clustering_model.pkl"

)


# ============================================================
# 11. CUSTOMER SEGMENT ANALYSIS
# ============================================================

segment_summary = df.groupby(
    "Customer_Cluster",
    observed=True
).agg(

    Customer_Count=("CustomerId", "count"),

    Average_Age=("Age", "mean"),

    Average_Balance=("Balance", "mean"),

    Average_Salary=("EstimatedSalary", "mean"),

    Average_Tenure=("Tenure", "mean"),

    Churn_Rate=("Exited", "mean"),

    Average_Value_Score=(
        "Customer_Value_Score",
        "mean"
    )

).reset_index()


segment_summary["Churn_Rate"] *= 100


print("\nCUSTOMER SEGMENT SUMMARY")

print(segment_summary)


segment_summary.to_csv(

    REPORT_PATH / "customer_segment_summary.csv",

    index=False

)


# ============================================================
# 12. CUSTOMER VALUE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(

    data=df,

    x="Value_Segment",

    order=[
        "Low Value",
        "Medium Value",
        "High Value",
        "Premium Value"
    ]

)

plt.title("Customer Value Distribution")

plt.xticks(rotation=20)

plt.savefig(

    FIGURE_PATH / "17_customer_value_distribution.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 13. CUSTOMER CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(

    data=df,

    x="Age",

    y="Balance",

    hue="Customer_Cluster",

    palette="viridis",

    alpha=0.7

)

plt.title("Customer Segmentation")

plt.savefig(

    FIGURE_PATH / "18_customer_segmentation.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 14. CLUSTER CHURN ANALYSIS
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(

    data=segment_summary,

    x="Customer_Cluster",

    y="Churn_Rate"

)

plt.title("Churn Rate by Customer Cluster")

plt.ylabel("Churn Rate (%)")

plt.savefig(

    FIGURE_PATH / "19_cluster_churn_analysis.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 15. SAVE CUSTOMER INTELLIGENCE DATA
# ============================================================

customer_output_columns = [

    "CustomerId",

    "Geography",

    "Gender",

    "Age",

    "Tenure",

    "Balance",

    "EstimatedSalary",

    "Exited",

    "Customer_Value_Score",

    "Value_Segment",

    "Customer_Cluster"

]

df[customer_output_columns].to_csv(

    OUTPUT_PATH / "customer_intelligence.csv",

    index=False

)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("STAGE 9 COMPLETED")

print("=" * 65)

print("\nTotal Customers:", len(df))

print("\nValue Segments:")

print(df["Value_Segment"].value_counts())

print("\nSelected Clusters:", best_k)

print("\nGenerated Reports:")

print("1. clustering_evaluation.csv")

print("2. customer_segment_summary.csv")

print("3. customer_intelligence.csv")

print("\nGenerated Visualizations:")

print("1. Customer Value Distribution")

print("2. Customer Segmentation")

print("3. Cluster Churn Analysis")

print("\nCustomer intelligence generated successfully!")
