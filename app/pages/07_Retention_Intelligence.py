import streamlit as st
import plotly.express as px
from utils.database import fetch_data

st.title("Retention Intelligence")
st.caption("Experimental email campaign uplift analysis")

df = fetch_data("""
SELECT treatment,conversion,uplift,uplift_category,dataset_source
FROM uplift_predictions
""")

if df.empty:
    st.warning("No uplift predictions available.")
    st.stop()

st.warning(
    "These predictions come from the Hillstrom email marketing experiment, "
    "not the bank customer dataset. Do not interpret them as validated "
    "bank retention campaign effects."
)

a,b,c = st.columns(3)
a.metric("Experiment Records",len(df))
b.metric("Treatment Customers",int(df.treatment.sum()))
c.metric("Average Predicted Uplift",f"{df.uplift.mean():.4f}")

summary = df.groupby("uplift_category",observed=False).agg(
    customers=("conversion","count"),
    average_uplift=("uplift","mean"),
    observed_conversion=("conversion","mean")
).reset_index()

st.subheader("Uplift Segment Analysis")
st.dataframe(summary,use_container_width=True)

st.plotly_chart(px.bar(summary,x="uplift_category",
    y="average_uplift",color="uplift_category"),
    use_container_width=True)

st.subheader("Predicted Uplift Distribution")
st.plotly_chart(px.histogram(df,x="uplift",nbins=40),
    use_container_width=True)

st.caption(
    "Individual treatment effects cannot be directly observed. "
    "Uplift is an estimated conditional treatment effect."
)\n