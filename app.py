import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Lifetime Value & Churn Intelligence")
st.caption("AI-powered Customer Analytics, Churn Prediction and Retention Intelligence")

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        try:
            database_url = st.secrets["DATABASE_URL"]
        except Exception:
            database_url = None

    if not database_url:
        raise ValueError("DATABASE_URL is missing.")

    return create_engine(database_url, pool_pre_ping=True)


@st.cache_data(ttl=60)
def get_data(query):
    with get_engine().connect() as connection:
        return pd.read_sql(text(query), connection)


def show_image(title, path):
    if Path(path).exists():
        st.subheader(title)
        st.image(path, use_container_width=True)
    else:
        st.info(f"{title} image is not available in the deployed repository.")


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

pages = [
    "Home",
    "Executive Dashboard",
    "Customer Intelligence",
    "Churn Prediction",
    "Explainable AI",
    "Survival Analysis",
    "Customer Segmentation",
    "Retention Intelligence",
    "ROI Simulator",
    "Model Performance"
]

selected_page = st.sidebar.selectbox(
    "Navigation",
    pages
)

st.sidebar.divider()
st.sidebar.caption("Customer Intelligence Platform")
st.sidebar.caption("Powered by Machine Learning & PostgreSQL")


# --------------------------------------------------
# LOAD DATABASE DATA
# --------------------------------------------------

try:
    customers = get_data("""
        SELECT * FROM customers
    """)

    predictions = get_data("""
        SELECT * FROM churn_predictions
    """)

    segments = get_data("""
        SELECT * FROM customer_segments
    """)

    uplift = get_data("""
        SELECT * FROM uplift_predictions
    """)

    metrics = get_data("""
        SELECT * FROM model_metrics
    """)

except Exception as error:
    st.error("Database connection failed.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# HOME
# --------------------------------------------------

if selected_page == "Home":

    st.header("Welcome to Customer Intelligence")

    st.markdown("""
    This platform combines customer analytics, machine learning,
    explainable AI and campaign intelligence to support
    data-driven business decisions.
    """)

    a, b, c = st.columns(3)

    a.metric("Total Customers", f"{len(customers):,}")
    b.metric("Churn Predictions", f"{len(predictions):,}")
    c.metric("Customer Segments", f"{len(segments):,}")

    st.divider()

    st.subheader("Platform Capabilities")

    st.markdown("""
    - Executive business intelligence
    - Customer profile exploration
    - Machine learning churn prediction
    - SHAP-based model explainability
    - Survival analysis feasibility
    - Customer value scoring and segmentation
    - Experimental uplift modelling
    - Retention campaign ROI simulation
    - Model performance evaluation
    """)

    st.success("PostgreSQL database connected successfully.")


# --------------------------------------------------
# EXECUTIVE DASHBOARD
# --------------------------------------------------

elif selected_page == "Executive Dashboard":

    st.header("Executive Dashboard")

    total = len(customers)
    churned = int(customers["exited"].sum())
    churn_rate = customers["exited"].mean() * 100

    avg_balance = customers["balance"].mean()
    avg_age = customers["age"].mean()

    a, b, c = st.columns(3)

    a.metric("Total Customers", f"{total:,}")
    b.metric("Churned Customers", f"{churned:,}")
    c.metric("Actual Churn Rate", f"{churn_rate:.2f}%")

    d, e, f = st.columns(3)

    d.metric("Average Balance", f"{avg_balance:,.2f}")
    e.metric("Average Age", f"{avg_age:.1f} years")
    f.metric("Predictions Generated", f"{len(predictions):,}")

    geo = customers.groupby("geography").agg(
        customers=("customer_id", "count"),
        churn_rate=("exited", "mean")
    ).reset_index()

    geo["churn_rate"] *= 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Geography")
        st.plotly_chart(
            px.bar(geo, x="geography", y="churn_rate",
                   color="geography"),
            use_container_width=True
        )

    with col2:
        st.subheader("Customer Distribution")
        st.plotly_chart(
            px.pie(geo, names="geography",
                   values="customers", hole=0.4),
            use_container_width=True
        )


# --------------------------------------------------
# CUSTOMER INTELLIGENCE
# --------------------------------------------------

elif selected_page == "Customer Intelligence":

    st.header("Customer Intelligence")

    search = st.text_input("Search Customer ID")

    df = customers.merge(
        segments,
        on="customer_id",
        how="left"
    )

    df = df.merge(
        predictions[["customer_id", "churn_probability"]],
        on="customer_id",
        how="left"
    )

    if search.strip():
        df = df[
            df["customer_id"].astype(str).str.contains(
                search.strip(), regex=False
            )
        ]

    geography = st.multiselect(
        "Filter by Geography",
        customers["geography"].dropna().unique().tolist(),
        default=customers["geography"].dropna().unique().tolist()
    )

    df = df[df["geography"].isin(geography)]

    st.metric("Matching Customers", len(df))
    st.dataframe(df, use_container_width=True)

    if not df.empty:

        selected = st.selectbox(
            "Select Customer",
            df["customer_id"].tolist()
        )

        customer = df[
            df["customer_id"] == selected
        ].iloc[0]

        st.subheader("Customer Profile")

        a, b, c = st.columns(3)

        a.metric("Age", int(customer["age"]))
        b.metric("Balance", f"{customer['balance']:,.2f}")
        c.metric("Tenure", f"{customer['tenure']} years")

        st.write("Geography:", customer["geography"])
        st.write("Customer Value Segment:",
                 customer.get("customer_value_segment", "N/A"))

        risk = customer.get("churn_probability")

        if pd.notna(risk):
            st.metric("Predicted Churn Risk", f"{risk:.2%}")


# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

elif selected_page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    df = predictions.merge(
        customers[["customer_id", "geography", "age", "balance"]],
        on="customer_id",
        how="left"
    )

    threshold = st.slider(
        "Churn Risk Threshold",
        0.0, 1.0, 0.5, 0.05
    )

    df["risk_category"] = df["churn_probability"].apply(
        lambda x: "High Risk" if x >= threshold else "Lower Risk"
    )

    a, b, c = st.columns(3)

    a.metric("Scored Customers", len(df))

    b.metric(
        "High Risk Customers",
        int((df["churn_probability"] >= threshold).sum())
    )

    c.metric(
        "Average Predicted Risk",
        f"{df['churn_probability'].mean():.2%}"
    )

    st.subheader("Churn Probability Distribution")

    st.plotly_chart(
        px.histogram(df, x="churn_probability",
                     color="risk_category", nbins=30),
        use_container_width=True
    )

    st.subheader("Highest Risk Customers")

    st.dataframe(
        df.sort_values(
            "churn_probability", ascending=False
        ).head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )


# --------------------------------------------------
# EXPLAINABLE AI
# --------------------------------------------------

elif selected_page == "Explainable AI":

    st.header("Explainable AI - SHAP")

    figures = [
        ("SHAP Summary", "reports/figures/14_shap_summary.png"),
        ("SHAP Feature Importance", "reports/figures/15_shap_feature_importance.png"),
        ("Customer Explanation", "reports/figures/16_customer_shap_waterfall.png"),
        ("Age Dependence", "reports/figures/shap_dependence_Age.png"),
        ("Balance Dependence", "reports/figures/shap_dependence_Balance.png"),
        ("Credit Score Dependence", "reports/figures/shap_dependence_CreditScore.png"),
        ("Tenure Dependence", "reports/figures/shap_dependence_Tenure.png")
    ]

    for title, path in figures:
        show_image(title, path)

    st.info(
        "SHAP explains how model features influence predictions. "
        "It does not establish causation."
    )


# --------------------------------------------------
# SURVIVAL ANALYSIS
# --------------------------------------------------

elif selected_page == "Survival Analysis":

    st.header("Survival Analysis")

    st.warning(
        "The current bank dataset does not contain reliable exit dates, "
        "observation dates or censoring information."
    )

    st.subheader("Available Tenure Information")

    survival = customers.groupby("exited").agg(
        customers=("customer_id", "count"),
        average_tenure=("tenure", "mean")
    ).reset_index()

    st.dataframe(survival, use_container_width=True)

    st.info("""
    A valid Kaplan-Meier or Cox model cannot be established from the
    available fields without unsupported assumptions. A suitable
    time-to-event dataset is needed.
    """)


# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

elif selected_page == "Customer Segmentation":

    st.header("Customer Value & Segmentation")

    df = customers.merge(
        segments,
        on="customer_id",
        how="inner"
    )

    a, b, c = st.columns(3)

    a.metric("Customers Segmented", len(df))
    b.metric("Value Segments",
             df["customer_value_segment"].nunique())
    c.metric("Clusters", df["cluster"].nunique())

    st.subheader("Customer Value Distribution")

    st.plotly_chart(
        px.histogram(
            df,
            x="customer_value_score",
            color="customer_value_segment",
            nbins=30
        ),
        use_container_width=True
    )

    summary = df.groupby(
        "customer_value_segment"
    ).agg(
        customers=("customer_id", "count"),
        average_balance=("balance", "mean"),
        churn_rate=("exited", "mean")
    ).reset_index()

    summary["churn_rate"] *= 100

    st.subheader("Segment Summary")
    st.dataframe(summary, use_container_width=True)

    st.plotly_chart(
        px.scatter(
            df,
            x="age",
            y="balance",
            color=df["cluster"].astype(str),
            hover_data=["customer_id", "geography"]
        ),
        use_container_width=True
    )

    st.caption(
        "Customer Value Score is a heuristic proxy, not actual "
        "financial Customer Lifetime Value."
    )


# --------------------------------------------------
# RETENTION INTELLIGENCE
# --------------------------------------------------

elif selected_page == "Retention Intelligence":

    st.header("Retention Intelligence")

    st.warning(
        "These predictions come from the Hillstrom email experiment, "
        "not from the bank customer dataset."
    )

    if uplift.empty:
        st.warning("No uplift records found.")
    else:

        a, b, c = st.columns(3)

        a.metric("Experiment Records", len(uplift))

        b.metric(
            "Treatment Customers",
            int(uplift["treatment"].sum())
        )

        c.metric(
            "Average Predicted Uplift",
            f"{uplift['uplift'].mean():.4f}"
        )

        summary = uplift.groupby(
            "uplift_category"
        ).agg(
            customers=("conversion", "count"),
            average_uplift=("uplift", "mean"),
            conversion_rate=("conversion", "mean")
        ).reset_index()

        st.subheader("Uplift Segment Analysis")
        st.dataframe(summary, use_container_width=True)

        st.plotly_chart(
            px.bar(
                summary,
                x="uplift_category",
                y="average_uplift",
                color="uplift_category"
            ),
            use_container_width=True
        )


# --------------------------------------------------
# ROI SIMULATOR
# --------------------------------------------------

elif selected_page == "ROI Simulator":

    st.header("Retention Campaign ROI Simulator")

    st.info(
        "This is an illustrative scenario calculator. "
        "Its inputs are assumptions, not measured campaign results."
    )

    target = st.number_input(
        "Customers Targeted", 1, 1000000, 1000
    )

    cost = st.number_input(
        "Campaign Cost per Customer (₹)",
        min_value=0.0, value=50.0
    )

    retention = st.slider(
        "Assumed Incremental Retention Rate (%)",
        0.0, 100.0, 5.0
    )

    value = st.number_input(
        "Assumed Value per Retained Customer (₹)",
        min_value=0.0, value=5000.0
    )

    total_cost = target * cost
    retained = target * retention / 100
    benefit = retained * value
    profit = benefit - total_cost

    roi = (
        profit / total_cost * 100
        if total_cost else 0
    )

    a, b, c = st.columns(3)

    a.metric("Campaign Cost", f"₹{total_cost:,.2f}")
    b.metric("Expected Retained Customers", f"{retained:,.0f}")
    c.metric("Estimated Net Benefit", f"₹{profit:,.2f}")

    st.metric("Illustrative ROI", f"{roi:.2f}%")

    if profit > 0:
        st.success("Positive ROI under these assumptions.")
    elif profit < 0:
        st.error("Negative ROI under these assumptions.")
    else:
        st.info("Break-even scenario.")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif selected_page == "Model Performance":

    st.header("Model Performance")

    if metrics.empty:
        st.warning("No evaluation metrics found.")
    else:

        st.subheader("Model Evaluation Metrics")

        st.dataframe(metrics, use_container_width=True)

        st.plotly_chart(
            px.bar(
                metrics,
                x="metric_name",
                y="metric_value",
                color="model_name",
                barmode="group"
            ),
            use_container_width=True
        )

    cv_path = Path(
        "reports/evaluation/cross_validation_results.csv"
    )

    if cv_path.exists():
        st.subheader("Cross Validation Results")
        st.dataframe(
            pd.read_csv(cv_path),
            use_container_width=True
        )

    show_image(
        "Confusion Matrix",
        "reports/figures/10_lightgbm_confusion_matrix.png"
    )

    show_image(
        "ROC Curve",
        "reports/figures/11_lightgbm_roc_curve.png"
    )

    show_image(
        "Precision Recall Curve",
        "reports/figures/12_lightgbm_pr_curve.png"
    )

    show_image(
        "Feature Importance",
        "reports/figures/13_feature_importance.png"
    )


st.sidebar.divider()
st.sidebar.caption("Customer Lifetime Value & Churn Intelligence")
st.sidebar.caption("Data Science Project | 2026")
