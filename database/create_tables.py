from sqlalchemy import text
from database.db_connection import engine

with open("database/schema.sql", "r") as file:
    schema = file.read()

statements = [
    statement.strip()
    for statement in schema.split(";")
    if statement.strip()
]

with engine.begin() as connection:
    for statement in statements:
        connection.execute(text(statement))

print("All five PostgreSQL tables created successfully!")

with engine.connect() as connection:
    result = connection.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """))

    print("\nAvailable tables:")
    for row in result:
        print("-", row[0])
