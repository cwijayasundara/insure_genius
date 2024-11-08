from llama_index.core import SQLDatabase, Settings
from llama_index.llms.openai import OpenAI
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import insert

def create_insurance_db():
    """
    Creates and populates an insurance member database, returns the query engine
    """
    Settings.llm = OpenAI("gpt-4o-mini")
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()

    member_table = Table(
        "member",
        metadata_obj,
        Column("member_id", String(16), primary_key=True),
        Column("member_name", String(50), nullable=False),
        Column("policy_type", String(50), nullable=False),
        Column("policy_number", String(50)),
        Column("last_claim_type", String(50)),
        Column("last_claim_amount", Integer),
    )

    metadata_obj.create_all(engine)

    rows = [
        {"member_id": "M001", "member_name": "John Doe", "policy_type": "Health", "policy_number": "P001", "last_claim_type": "Accident", "last_claim_amount": 5000},
        {"member_id": "M002", "member_name": "Jane Smith", "policy_type": "Life", "policy_number": "P002", "last_claim_type": "Critical Illness", "last_claim_amount": 2000},
        {"member_id": "M003", "member_name": "Alice Johnson", "policy_type": "Auto", "policy_number": "P003", "last_claim_type": "Collision", "last_claim_amount": 1500},
        {"member_id": "M004", "member_name": "Bob Brown", "policy_type": "Home", "policy_number": "P004", "last_claim_type": "Fire", "last_claim_amount": 10000},
        {"member_id": "M005", "member_name": "Charlie Davis", "policy_type": "Travel", "policy_number": "P005", "last_claim_type": "Trip Cancellation", "last_claim_amount": 3000},
    ]

    for row in rows:
        stmt = insert(member_table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)

    sql_database = SQLDatabase(engine, include_tables=["member"])
    
    return NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["member"]
    )

# Example usage:
# if __name__ == "__main__":
#     query_engine = create_insurance_db()
#     query = "What is the policy number for the member with the name John Doe?"
#     response = query_engine.query(query)
#     print(response)