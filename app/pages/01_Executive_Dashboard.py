import streamlit as st
import plotly.express as px
from utils.database import fetch_data

st.title("Executive Dashboard")
st.caption("Business overview of customer churn and intelligence")

k = fetch_data("""
SELECT COUNT(*) AS total,
       SUM(exited) AS churned,
       ROUND(AVG(exited::numeric)*100,2) AS churn_rate,
       ROUND(AVG(balance),2) AS avg_balance
FROM customers
""").iloc[0]

a,b,c,d = st.columns(4)
a.metric("Total Customers", f"{int(k.total):,}")
b.metric("Churned Customers", f"{int(k.churned):,}")
c.metric("Actual Churn Rate", f"{k.churn_rate}%")
d.metric("Average Balance", f"{k.avg_balance:,.2f}")

st.divider()

geo = fetch_data("""
SELECT geography, COUNT(*) AS customers,
ROUND(AVG(exited::numeric)*100,2) AS churn_rate
FROM customers GROUP BY geography
""")

col1,col2 = st.columns(2)

with col1:
    st.subheader("Churn by Geography")
    st.plotly_chart(px.bar(geo,x="geography",y="churn_rate",
        color="geography",text="churn_rate"),use_container_width=True)

with col2:
    st.subheader("Customer Distribution")
    st.plotly_chart(px.pie(geo,names="geography",values="customers",
        hole=.4),use_container_width=True)

st.subheader("Customer Value Segments")
segments = fetch_data("""
SELECT customer_value_segment, COUNT(*) AS customers
FROM customer_segments
GROUP BY customer_value_segment
""")

st.plotly_chart(px.bar(segments,x="customer_value_segment",
    y="customers",color="customer_value_segment"),
    use_container_width=True)\n