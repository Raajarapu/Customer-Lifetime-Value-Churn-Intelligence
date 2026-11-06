import pandas as pd
import joblib

from sqlalchemy import text
from database.db_connection import engine

print("Starting PostgreSQL data loading...")


# --------------------------------------------------
# 1. LOAD CUSTOMER DATA
# --------------------------------------------------

customers = pd.read_csv(
    "data/raw/Customer-Churn-Records.csv"
)

customers = customers.rename(columns={
    "CustomerId": "customer_id",
    "Satisfaction Score": "satisfaction_score",
    "Card Type": "card_type",
    "Point Earned": "point_earned"
})

customer_columns = [
    "customer_id", "Surname", "CreditScore", "Geography",
    "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Exited", "Complain", "satisfaction_score",
    "card_type", "point_earned"
]

customers = customers[customer_columns].copy()

customers.columns = [
    "customer_id", "surname", "credit_score", "geography",
    "gender", "age", "tenure", "balance", "num_of_products",
    "has_cr_card", "is_active_member", "estimated_salary",
    "exited", "complain", "satisfaction_score",
    "card_type", "point_earned"
]


# --------------------------------------------------
# 2. LOAD CHURN PREDICTIONS
# --------------------------------------------------

X_test = pd.read_csv("data/processed/X_test.csv")
customer_ids = pd.read_csv(
    "data/processed/customer_id_test.csv"
)

churn_model = joblib.load(
    "models/lightgbm_churn_model.pkl"
)

churn_probabilities = churn_model.predict_proba(X_test)[:, 1]

churn_predictions = pd.DataFrame({
    "customer_id": customer_ids["CustomerId"],
    "predicted_churn": (churn_probabilities >= 0.5).astype(int),
    "churn_probability": churn_probabilities,
    "model_name": "LightGBM"
})


# --------------------------------------------------
# 3. LOAD CUSTOMER SEGMENTS
# --------------------------------------------------

segments = pd.read_csv(
    "data/processed/customer_intelligence.csv"
)

segments = segments.rename(columns={
    "CustomerId": "customer_id",
    "Customer_Value_Score": "customer_value_score",
    "Value_Segment": "customer_value_segment",
    "Customer_Cluster": "cluster"
})

segments = segments[[
    "customer_id",
    "customer_value_score",
    "customer_value_segment",
    "cluster"
]]


# --------------------------------------------------
# 4. LOAD UPLIFT PREDICTIONS
# --------------------------------------------------

uplift = pd.read_csv(
    "data/processed/uplift_predictions.csv"
)

uplift["dataset_source"] = "Hillstrom Email Campaign"


# --------------------------------------------------
# 5. LOAD MODEL METRICS
# --------------------------------------------------

evaluation = pd.read_csv(
    "reports/evaluation/model_evaluation.csv"
)

evaluation = evaluation.rename(
    columns={"Unnamed: 0": "model_name"}
)

metric_columns = [
    column for column in evaluation.columns
    if column != "model_name"
]

metrics = evaluation.melt(
    id_vars="model_name",
    value_vars=metric_columns,
    var_name="metric_name",
    value_name="metric_value"
)

metrics = metrics.dropna()


# --------------------------------------------------
# 6. INSERT DATA INTO POSTGRESQL
# --------------------------------------------------

tables = {
    "customers": customers,
    "churn_predictions": churn_predictions,
    "customer_segments": segments,
    "uplift_predictions": uplift,
    "model_metrics": metrics
}

with engine.begin() as connection:

    # Make this loader safe to rerun without duplicating records.
    for table in tables:
        connection.execute(
            text(f"TRUNCATE TABLE {table} RESTART IDENTITY")
        )

    for table, dataframe in tables.items():
        dataframe.to_sql(
            table,
            con=connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000
        )

        print(f"{table}: {len(dataframe)} records inserted")


# --------------------------------------------------
# 7. VERIFY DATABASE RECORD COUNTS
# --------------------------------------------------

print("\nDATABASE VERIFICATION")

with engine.connect() as connection:

    for table in tables:
        count = connection.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar_one()

        print(f"{table}: {count} records")


print("\nSTAGE 11 DATA LOADING COMPLETED!")
