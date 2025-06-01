import os
import openai
import psycopg2
import pgvector.psycopg2

# API キーは環境変数から
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="password123"
)

texts = ["りんご", "バナナ", "果物", "渋谷", "新宿", "東京都"]

with conn:
    with conn.cursor() as cur:
        for text in texts:
            emb = get_embedding(text)
            cur.execute(
                "INSERT INTO items (content, embedding) VALUES (%s, %s)",
                (text, emb)
            )
print("埋め込み挿入完了！")
