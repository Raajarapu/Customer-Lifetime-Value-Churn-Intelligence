import streamlit as st
import pandas as pd

st.title("Retention Campaign ROI Simulator")
st.caption("Illustrative business scenario calculator")

st.info(
    "This simulator uses user-defined assumptions. It is not a measured "
    "ROI result from the bank dataset or Hillstrom experiment."
)

customers = st.number_input(
    "Customers targeted",min_value=1,value=1000,step=100
)

campaign_cost = st.number_input(
    "Campaign cost per customer (₹)",min_value=0.0,value=50.0
)

retention_rate = st.slider(
    "Assumed incremental retention rate (%)",0.0,100.0,5.0
)

customer_value = st.number_input(
    "Assumed value per retained customer (₹)",
    min_value=0.0,value=5000.0
)

a,b,c = st.columns(3)

total_cost = customers*campaign_cost
retained = customers*retention_rate/100
benefit = retained*customer_value
profit = benefit-total_cost
roi = (profit/total_cost*100) if total_cost else 0

a.metric("Campaign Cost",f"₹{total_cost:,.2f}")
b.metric("Expected Retained Customers",f"{retained:,.0f}")
c.metric("Estimated Net Benefit",f"₹{profit:,.2f}")

st.metric("Illustrative ROI",f"{roi:.2f}%")

if profit > 0:
    st.success("Positive ROI under the selected assumptions.")
elif profit < 0:
    st.error("Negative ROI under the selected assumptions.")
else:
    st.info("Break-even scenario.")

st.caption(
    "All inputs are assumptions. Replace them with validated campaign "
    "costs, incremental retention estimates and customer contribution "
    "margin before making business decisions."
)\n