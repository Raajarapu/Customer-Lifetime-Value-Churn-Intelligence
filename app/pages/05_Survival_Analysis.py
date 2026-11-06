import streamlit as st
import pandas as pd
from utils.database import fetch_data

st.title("Survival Analysis")
st.caption("Time-to-event analysis feasibility")

st.warning(
    "The current bank dataset does not contain reliable customer exit dates "
    "or observation/censoring dates. Therefore, a valid Kaplan-Meier curve "
    "or Cox proportional hazards model cannot currently be estimated."
)

df = fetch_data("""
SELECT exited, COUNT(*) AS customers,
AVG(tenure) AS average_tenure
FROM customers GROUP BY exited ORDER BY exited
""")

st.subheader("Available Tenure Information")
st.dataframe(df,use_container_width=True)

st.subheader("Why survival modelling is not yet valid")

st.markdown("""
- Tenure is available, but its exact time origin and relationship to churn
  are not sufficiently documented.
- Customer exit dates are unavailable.
- Censoring status and observation dates are unavailable.
- Treating tenure as observed survival duration would introduce unsupported
  assumptions.
""")

st.success(
    "Feasibility assessment completed. A separate dataset with valid "
    "duration and event/censoring information is required for survival modelling."
)\n