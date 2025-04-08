import mysql.connector
from sentence_transformers import SentenceTransformer
import json
from decouple import config

# Loading model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

DB_CONFIG = {
    "host": config('db_host'),
    "user": config('db_user'),
    "password": config('db_password'),
    "database": config('db_name'),
    "port": config('db_port'),
    "charset": 'utf8mb4'
}

# Connection to DB
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Get all questions without embeddings
cursor.execute("SELECT id, question FROM question_answer WHERE embedding IS NULL")
rows = cursor.fetchall()

print(f"Total questions without embeddings: {len(rows)}")

for row_id, question in rows:
    embedding = model.encode(question)
    # Save as Json
    embedding_json = json.dumps(embedding.tolist())

    cursor.execute(
        "UPDATE question_answer SET embedding = %s WHERE id = %s",
        (embedding_json, row_id)
    )

conn.commit()
cursor.close()
conn.close()

print("Done: embeddings saved.")
