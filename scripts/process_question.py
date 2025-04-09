import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
from decouple import config
from langdetect import detect
import json

# DB settings
DB_CONFIG = {
    "host": config('db_host'),
    "user": config('db_user'),
    "password": config('db_password'),
    "database": config('db_name'),
    "port": config('db_port'),
    "charset": 'utf8mb4'
}
openai.api_key = config('openai_api_key')

# Model initialization
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Generation text by GPT
def call_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Main logic
def process_question(user_question: str) -> str:
    # Getting question embedding
    question_embedding = model.encode(user_question)

    # Connection to DB
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    # Download all embeddings from question_answer
    cursor.execute("SELECT id, answer, embedding FROM question_answer WHERE processed IS True")
    rows = cursor.fetchall()

    if not rows:
        return "DB is empty."

    # Comparing embeddings
    similarities = []
    for row in rows:
        try:
            db_vector = np.array(json.loads(row['embedding']))
            score = util.cos_sim(question_embedding, db_vector).item()
            similarities.append((score, row))
        except Exception as e:
            print(f"Error on processing ID row {row['id']}: {e}")
            continue

    # Descenders sort
    similarities.sort(reverse=True, key=lambda x: x[0])

    if similarities and similarities[0][0] >= 0.7:
        top_matches = similarities[:2]
        responses = [match[1]['answer'] for match in top_matches]
        return "Vezi ce am gasit pentru întrebarea ta:\n\n" + "\n\n".join(responses)


    # Else - GPT
    gpt_response = call_gpt(user_question)
    language = detect_language(gpt_response)

    cursor.execute("""
        INSERT INTO question_answer (question, answer, language, embedding, processed, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """, (user_question, gpt_response, language, str(question_embedding.tolist()), False))

    conn.commit()
    cursor.close()
    conn.close()

    return gpt_response

def detect_language(text: str) -> str:
    lang = detect(text)
    if lang.startswith('ru'):
        return 'ru'
    elif lang.startswith('ro'):
        return 'ro'
    elif lang.startswith('en'):
        return 'eng'
    else:
        return 'ru' #default value

# === Example ===
if __name__ == "__main__":
    question = "Какие документы нужны ребёнку для поездки за границу?"
    print(process_question(question))
