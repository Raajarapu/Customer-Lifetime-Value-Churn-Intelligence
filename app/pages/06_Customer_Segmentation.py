import streamlit as st
import plotly.express as px
from utils.database import fetch_data

st.title("Customer Value & Segmentation")
st.caption("Heuristic customer value scoring and K-Means clusters")

df = fetch_data("""
SELECT c.customer_id,c.age,c.balance,c.geography,c.exited,
s.customer_value_score,s.customer_value_segment,s.cluster
FROM customers c
JOIN customer_segments s ON c.customer_id=s.customer_id
""")

a,b,c = st.columns(3)
a.metric("Customers Segmented",len(df))
b.metric("Value Segments",df.customer_value_segment.nunique())
c.metric("Clusters",df.cluster.nunique())

st.subheader("Customer Value Distribution")
st.plotly_chart(px.histogram(df,x="customer_value_score",
    color="customer_value_segment",nbins=30),
    use_container_width=True)

st.subheader("Value Segment Distribution")
segment = df.groupby("customer_value_segment").agg(
    customers=("customer_id","count"),
    average_balance=("balance","mean"),
    churn_rate=("exited","mean")
).reset_index()

segment["churn_rate"] *= 100

col1,col2 = st.columns(2)

with col1:
    st.plotly_chart(px.pie(segment,names="customer_value_segment",
        values="customers",hole=.4),use_container_width=True)

with col2:
    st.plotly_chart(px.bar(segment,x="customer_value_segment",
        y="churn_rate",color="customer_value_segment"),
        use_container_width=True)

st.subheader("Customer Cluster Analysis")
st.plotly_chart(px.scatter(df,x="age",y="balance",
    color=df["cluster"].astype(str),size="customer_value_score",
    hover_data=["customer_id","geography"]),
    use_container_width=True)

st.dataframe(segment,use_container_width=True)

st.caption(
    "Customer Value Score is a heuristic proxy, not actual financial CLV. "
    "The dataset has no historical revenue or profit data."
)\n