import streamlit as st
import plotly.express as px

from utils.database import fetch_data

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Lifetime Value & Churn Intelligence")
st.markdown(
    "### AI-powered Customer Analytics and Retention Intelligence"
)

st.divider()

try:
    # Customer KPIs
    customer_kpis = fetch_data("""
        SELECT
            COUNT(*) AS total_customers,
            SUM(exited) AS churned_customers,
            ROUND(AVG(exited::numeric) * 100, 2) AS churn_rate,
            ROUND(AVG(balance), 2) AS average_balance,
            ROUND(AVG(age), 1) AS average_age
        FROM customers
    """)

    churn_kpis = fetch_data("""
        SELECT
            ROUND(AVG(churn_probability) * 100, 2)
            AS average_predicted_churn
        FROM churn_predictions
    """)

    segments = fetch_data("""
        SELECT
            customer_value_segment,
            COUNT(*) AS customer_count
        FROM customer_segments
        GROUP BY customer_value_segment
        ORDER BY customer_count DESC
    """)

    k = customer_kpis.iloc[0]
    predicted = churn_kpis.iloc[0]

    # KPI cards
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{int(k['total_customers']):,}"
    )

    col2.metric(
        "Actual Churn Rate",
        f"{k['churn_rate']}%"
    )

    col3.metric(
        "Churned Customers",
        f"{int(k['churned_customers']):,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Average Customer Balance",
        f"₹{k['average_balance']:,.2f}"
    )

    col5.metric(
        "Average Customer Age",
        f"{k['average_age']} years"
    )

    col6.metric(
        "Average Predicted Churn Risk",
        f"{predicted['average_predicted_churn']}%"
    )

    st.divider()

    # Customer segmentation chart
    st.subheader("Customer Value Segmentation")

    fig = px.pie(
        segments,
        names="customer_value_segment",
        values="customer_count",
        hole=0.45,
        title="Distribution of Customer Value Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Customer information
    st.subheader("Platform Overview")

    st.markdown("""
    This platform integrates:

    - Machine Learning-based churn prediction.
    - SHAP-based model explainability.
    - Customer value scoring and segmentation.
    - Experimental uplift modelling.
    - PostgreSQL-based customer intelligence.
    - Business-oriented retention analytics.
    """)

    st.success("Connected successfully to PostgreSQL!")

except Exception as error:
    st.error("Unable to load dashboard data.")
    st.exception(error)
