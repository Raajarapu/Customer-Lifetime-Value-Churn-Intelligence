import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Explainable AI")
st.caption("Understanding the factors behind model predictions")

st.subheader("SHAP Global Feature Importance")

figures = [
    ("SHAP Summary","reports/figures/14_shap_summary.png"),
    ("SHAP Feature Importance","reports/figures/15_shap_feature_importance.png"),
    ("Customer Explanation","reports/figures/16_customer_shap_waterfall.png"),
    ("Age Dependence","reports/figures/shap_dependence_Age.png"),
    ("Balance Dependence","reports/figures/shap_dependence_Balance.png"),
    ("Credit Score Dependence","reports/figures/shap_dependence_CreditScore.png"),
    ("Tenure Dependence","reports/figures/shap_dependence_Tenure.png")
]

for title,path in figures:
    if Path(path).exists():
        with st.expander(title,expanded=(title=="SHAP Summary")):
            st.image(path,use_container_width=True)

csv_path = Path("reports/evaluation/customer_shap_explanation.csv")

if csv_path.exists():
    st.subheader("Customer SHAP Explanation Data")
    explanation = pd.read_csv(csv_path)
    st.dataframe(explanation.head(100),use_container_width=True)
else:
    st.info("Customer-level SHAP explanation file is not available.")

st.info(
    "SHAP values explain model output. They describe model behaviour, "
    "not proof of causation."
)\n