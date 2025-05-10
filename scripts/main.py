from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from decouple import config
from langdetect import detect
import json

app = FastAPI()

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

class QuestionRequest(BaseModel):
    question: str

def get_relevant_answers() -> list[tuple[str, str]]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, answer FROM question_answer 
            WHERE language = 'eng' AND id > 6 AND id < 44
        """)
        return cursor.fetchall()
    except Exception as e:
        print(f"DB error when fetching examples: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# Generation text by GPT
def call_gpt(prompt: str) -> str:
    examples_text = get_relevant_answers()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant specializing in questions strictly related to crossing the border of the Republic of Moldova. "
                "You must only answer questions directly related to this topic, such as required documents, border crossing rules, categories of travelers and documents, control procedures. "
                "Do not answer any questions that are outside the scope of border crossing.\n\n"
                "If a question is unrelated or unclear, politely refuse to answer and inform the user that they can contact the Border Police of the Republic of Moldova for clarification "
                "via phone at +37322259717 or by email at linia.verde@border.gov.md.\n\n"
                "In case of doubt, always prioritize not answering over providing potentially incorrect or off-topic information.\n\n"
                "Keep all responses clear, professional and directly related to the topic. Do not mention that you are an AI language model."
            )
        }
    ]

    for example in examples_text:
        q, a = example
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    # finally add user question
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,  # reduce creativity so as not to go off-topic
        top_p=1.0,
        max_tokens = 500 # about 400 words
    )
    gpt_response = response.choices[0].message.content.strip()
    print("in call gpt - " + gpt_response + "\n\n")

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

@app.post("/ask")
def process_question(req: QuestionRequest):
    user_question = req.question

    # 1. Getting question embedding
    try:
        question_embedding = model.encode(user_question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        # 2. Connecting to DB
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # 3. Fetching data from DB
        cursor.execute("SELECT id, answer, embedding FROM question_answer WHERE processed IS True")
        rows = cursor.fetchall()

        if not rows:
            return {"answer": "DB is empty."}

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

        if similarities and similarities[0][0] >= 0.9:
            best_match = similarities[0][1]
            return {"answer": best_match['answer']}

        # Else - GPT and saving new question and answer to DB
        gpt_response = call_gpt(user_question)
        language = detect_language(gpt_response)

        cursor.execute("""
            INSERT INTO question_answer (question, answer, language, embedding, processed, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        """, (user_question, gpt_response, language, str(question_embedding.tolist()), False))

        conn.commit()
        print("in process_question - " + gpt_response + "\n\n")
        print(repr(gpt_response))
        return {"answer": gpt_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()