import os
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

@st.cache_resource
def get_engine():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        try:
            database_url = st.secrets["DATABASE_URL"]
        except Exception:
            database_url = None

    if not database_url:
        raise ValueError("DATABASE_URL not found in .env or Streamlit secrets.")

    return create_engine(
        database_url,
        pool_pre_ping=True
    )


def fetch_data(query, params=None):

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(
            text(query),
            connection,
            params=params
        )
