import time
import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
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

client = OpenAI(api_key=config('openai_api_key'))

# Encoder Model (for embeddings) initialization
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Generation text by GPT
def call_gpt(prompt: str) -> str:
    system_prompt = (
        "You are an assistant specializing in questions strictly related to crossing the border of the Republic of Moldova. "
        "You must only answer questions directly related to this topic, such as required documents, border crossing rules, categories of travelers, control procedures, or emergency situations. "
        "Do not answer any questions that are outside the scope of border crossing, "
        "If a question is unrelated or unclear, politely refuse to answer and inform the user that they can contact the Border Police of the Republic of Moldova for clarification "
        "via phone at +37322259717 or by email at linia.verde@border.gov.md. "
        "In case of doubt, always prioritize not answering over providing potentially incorrect or off-topic information. "
        "Keep all responses short, clear, professional and under 100 words"
        "Do not mention that you are an AI language model."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,  # reduce creativity so as not to go off-topic
        top_p=0.5,
        max_tokens = 150 # about 100-110 words
    )
    return response.choices[0].message.content.strip()

# Main logic
def process_question(user_question: str) -> str:
    # 1. Getting question embedding
    question_embedding = model.encode(user_question)

    # 2. Connecting to DB
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    # 3. Fetching data from DB
    cursor.execute("SELECT id, answer, embedding FROM question_answer WHERE processed IS True")
    rows = cursor.fetchall()

    if not rows:
        return "DB is empty."

    # 4. Vector comparison
    similarities = []
    for row in rows:
        try:
            db_vector = np.array(json.loads(row['embedding']), dtype=np.float32)
            question_vector = np.array(question_embedding, dtype=np.float32)
            score = util.cos_sim(question_vector, db_vector).item()
            similarities.append((score, row))
        except Exception as e:
            print(f"Error on processing ID row {row['id']}: {e}")
            continue

    # 5. Sorting and picking top result
    similarities.sort(reverse=True, key=lambda x: x[0])

    if similarities and similarities[0][0] >= 0.7:
        best_match = similarities[0][1]
        return best_match['answer']

    # Else - GPT and saving new question and answer to DB
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
    question1 = "Какие документы нужны ребёнку для поездки за границу?"
    question2 = "How can I calculate the period of my staying in the Republic of Moldova"
    print(process_question(question1))
