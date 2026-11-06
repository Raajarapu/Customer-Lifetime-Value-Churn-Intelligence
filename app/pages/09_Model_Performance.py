import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from utils.database import fetch_data

st.title("Model Performance")
st.caption("Evaluation of the customer churn prediction model")

metrics = fetch_data("""
SELECT model_name,metric_name,metric_value
FROM model_metrics
ORDER BY model_name,metric_name
""")

if metrics.empty:
    st.warning("No model metrics are available.")
else:
    st.subheader("Evaluation Metrics")
    st.dataframe(metrics,use_container_width=True)

    st.plotly_chart(px.bar(
        metrics,
        x="metric_name",
        y="metric_value",
        color="model_name",
        barmode="group",
        title="Model Evaluation Comparison"
    ),use_container_width=True)

cv_path = Path("reports/evaluation/cross_validation_results.csv")

if cv_path.exists():
    st.subheader("Cross-Validation Results")
    cv = pd.read_csv(cv_path)
    st.dataframe(cv,use_container_width=True)

for title,path in [
    ("Confusion Matrix","reports/figures/10_lightgbm_confusion_matrix.png"),
    ("ROC Curve","reports/figures/11_lightgbm_roc_curve.png"),
    ("Precision-Recall Curve","reports/figures/12_lightgbm_pr_curve.png"),
    ("Feature Importance","reports/figures/13_feature_importance.png")
]:
    from pathlib import Path
    if Path(path).exists():
        with st.expander(title):
            st.image(path,use_container_width=True)\n