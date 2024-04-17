import psycopg2
import psycopg2.extras
import os

conn = psycopg2.connect( database=os.getenv("PGVECTOR_DATABASE"),
                            host=os.getenv("PGVECTOR_HOST"),
                            user=os.getenv("PGVECTOR_USER"),
                            password=os.getenv("PGVECTOR_PASSWORD"),
                            port=int(os.getenv("PGVECTOR_PORT")))
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)


def get_lab_collections():
    query = "SELECT name from langchain_pg_collection WHERE name like 'LAB_%'"
    cursor.execute(query)
    result = cursor.fetchall()
    return result

def get_documents_by_collection_name(name):
    query = f"""SELECT document 
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c 
                ON e.collection_id = c.uuid 
                WHERE c.name = '{name}'"""
    cursor.execute(query)
    result = cursor.fetchall()
    return result
        
    