import streamlit as st
from utils.database import fetch_data

st.title("Customer Intelligence")
st.caption("Explore individual customer profiles and behaviour")

search = st.text_input("Search by Customer ID")

geo = st.multiselect(
    "Geography",
    ["France","Germany","Spain"],
    default=["France","Germany","Spain"]
)

if not geo:
    st.warning("Select at least one geography.")
    st.stop()

params = {"geographies": geo}

query = """
SELECT c.customer_id, c.geography, c.gender, c.age,
c.tenure, c.balance, c.num_of_products, c.is_active_member,
c.exited, s.customer_value_score, s.customer_value_segment,
s.cluster, p.churn_probability
FROM customers c
LEFT JOIN customer_segments s ON c.customer_id=s.customer_id
LEFT JOIN churn_predictions p ON c.customer_id=p.customer_id
WHERE c.geography = ANY(:geographies)
"""

if search.strip():
    query += " AND CAST(c.customer_id AS TEXT) LIKE :customer_id"
    params["customer_id"] = f"%{search.strip()}%"

df = fetch_data(query,params)

st.metric("Matching Customers",len(df))
st.dataframe(df,use_container_width=True)

if not df.empty:
    selected = st.selectbox("Select Customer",df["customer_id"].tolist())
    customer = df[df["customer_id"]==selected].iloc[0]

    st.subheader(f"Customer Profile: {selected}")
    a,b,c = st.columns(3)
    a.metric("Age",int(customer.age))
    b.metric("Balance",f"{customer.balance:,.2f}")
    c.metric("Value Segment",str(customer.customer_value_segment))

    if customer.churn_probability is not None:
        st.metric("Predicted Churn Probability",
                  f"{customer.churn_probability:.2%}")
    else:
        st.info("No saved model prediction for this customer.")\n