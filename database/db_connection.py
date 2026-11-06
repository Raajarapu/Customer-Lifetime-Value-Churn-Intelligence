import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL missing from .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

if __name__ == "__main__":
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT current_database(), current_user")
        )
        database, user = result.fetchone()

        print("PostgreSQL connection successful!")
        print("Database:", database)
        print("User:", user)
