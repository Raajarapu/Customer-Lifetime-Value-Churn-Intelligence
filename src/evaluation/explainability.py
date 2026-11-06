# ============================================================
# CUSTOMER LIFETIME VALUE AND CHURN INTELLIGENCE SYSTEM
# STAGE 7: EXPLAINABLE AI USING SHAP
# ============================================================

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed"

MODEL_PATH = PROJECT_ROOT / "models"

FIGURE_PATH = PROJECT_ROOT / "reports/figures"

REPORT_PATH = PROJECT_ROOT / "reports/evaluation"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("STAGE 7: SHAP EXPLAINABILITY")
print("=" * 65)


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

model = joblib.load(
    MODEL_PATH / "lightgbm_churn_model.pkl"
)

X_train = pd.read_csv(
    DATA_PATH / "X_train.csv"
)

X_test = pd.read_csv(
    DATA_PATH / "X_test.csv"
)

print("\nModel and datasets loaded successfully.")

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ============================================================
# 3. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

# Use a representative sample to control computation time.
X_explain = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

shap_values = explainer(X_explain)

print("\nSHAP Values Generated Successfully!")


# ============================================================
# 4. SHAP SUMMARY PLOT
# ============================================================

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_explain,
    show=False
)

plt.title("SHAP Summary Plot")

plt.savefig(
    FIGURE_PATH / "14_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. SHAP FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(10, 7))

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title("SHAP Feature Importance")

plt.savefig(
    FIGURE_PATH / "15_shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. SHAP DEPENDENCE PLOTS
# ============================================================

important_features = [
    "Age",
    "Balance",
    "CreditScore",
    "Tenure"
]

for feature in important_features:

    if feature not in X_explain.columns:
        print(f"Skipping {feature}: Feature not found.")
        continue

    print(f"Generating SHAP dependence plot: {feature}")

    plt.figure(figsize=(10, 6))

    shap.dependence_plot(
        feature,
        shap_values.values,
        X_explain,
        interaction_index="auto",
        show=False
    )

    plt.title(f"SHAP Dependence Plot - {feature}")

    plt.savefig(
        FIGURE_PATH / f"shap_dependence_{feature}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: shap_dependence_{feature}.png")


# ============================================================
# 7. INDIVIDUAL CUSTOMER EXPLANATION
# ============================================================

customer_index = 0

customer_explanation = shap_values[customer_index]

plt.figure(figsize=(12, 7))

shap.plots.waterfall(
    customer_explanation,
    max_display=15,
    show=False
)

plt.title("Individual Customer Churn Explanation")

plt.savefig(
    FIGURE_PATH / "16_customer_shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. CUSTOMER LEVEL EXPLANATION REPORT
# ============================================================

values = customer_explanation.values

feature_names = X_explain.columns

explanation_report = pd.DataFrame({

    "Feature": feature_names,

    "Feature_Value": X_explain.iloc[customer_index].values,

    "SHAP_Value": values

})

explanation_report["Impact"] = np.where(
    explanation_report["SHAP_Value"] > 0,
    "Increases Churn Risk",
    "Decreases Churn Risk"
)

explanation_report["Absolute_Impact"] = (
    explanation_report["SHAP_Value"].abs()
)

explanation_report = explanation_report.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\nCUSTOMER EXPLANATION")

print(explanation_report.head(15))


explanation_report.to_csv(
    REPORT_PATH / "customer_shap_explanation.csv",
    index=False
)


# ============================================================
# 9. SAVE SHAP VALUES
# ============================================================

shap_report = pd.DataFrame(
    shap_values.values,
    columns=X_explain.columns
)

shap_report.to_csv(
    REPORT_PATH / "shap_values.csv",
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\nGenerated SHAP Reports:")

print("1. SHAP Summary Plot")
print("2. SHAP Feature Importance")
print("3. Feature Dependence Plots")
print("4. Individual Customer Waterfall")
print("5. Customer Explanation CSV")
print("6. SHAP Values CSV")

print("\nSTAGE 7 COMPLETED SUCCESSFULLY!")

print("=" * 65)