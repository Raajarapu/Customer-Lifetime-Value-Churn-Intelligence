import streamlit as st
import plotly.express as px
from utils.database import fetch_data

st.title("Churn Prediction")
st.caption("Explore LightGBM churn risk predictions")

df = fetch_data("""
SELECT p.customer_id,p.predicted_churn,p.churn_probability,
c.geography,c.age,c.balance,c.is_active_member
FROM churn_predictions p
JOIN customers c ON p.customer_id=c.customer_id
""")

if df.empty:
    st.warning("No churn predictions available.")
    st.stop()

threshold = st.slider("Churn risk threshold",0.0,1.0,0.5,0.05)

df["risk_category"] = df["churn_probability"].apply(
    lambda x: "High Risk" if x >= threshold else "Lower Risk"
)

a,b,c = st.columns(3)
a.metric("Customers Scored",len(df))
b.metric("High-Risk Customers",int((df.churn_probability>=threshold).sum()))
c.metric("Average Risk",f"{df.churn_probability.mean():.2%}")

st.subheader("Churn Probability Distribution")
st.plotly_chart(px.histogram(df,x="churn_probability",nbins=30,
    color="risk_category"),use_container_width=True)

st.subheader("Highest-Risk Customers")
st.dataframe(
    df.sort_values("churn_probability",ascending=False).head(100),
    use_container_width=True
)

st.download_button(
    "Download Churn Predictions",
    df.to_csv(index=False),
    "churn_predictions.csv",
    "text/csv"
)\n