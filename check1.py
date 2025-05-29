import os
import openai
import psycopg2
import numpy as np

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

# 検索クエリ
query = "食べ物"
query_embedding = get_embedding(query)

with conn:
    # まず、pgvectorが正しくインストールされていることを確認
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    
    with conn.cursor() as cur:
        # 方法1: 配列を直接SQLに埋め込む
        embedding_array = np.array(query_embedding).tolist()
        sql = """
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity 
        FROM items 
        ORDER BY embedding <=> %s::vector 
        LIMIT 3
        """
        
        # ベクトルを文字列として正しく構築
        vector_str = f"[{','.join(str(x) for x in embedding_array)}]"
        
        try:
            cur.execute(sql, (vector_str, vector_str))
            results = cur.fetchall()
            
            print(f"「{query}」に類似したアイテム:")
            for content, similarity in results:
                print(f"コンテンツ: {content}, 類似度: {similarity}")
        
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            
            # 方法2: vector_ip関数を使用する場合
            try:
                sql_alt = """
                SELECT content, vector_ip(embedding, %s::vector) AS similarity 
                FROM items 
                ORDER BY vector_ip(embedding, %s::vector) DESC
                LIMIT 3
                """
                cur.execute(sql_alt, (vector_str, vector_str))
                results = cur.fetchall()
                
                print(f"「{query}」に類似したアイテム (方法2):")
                for content, similarity in results:
                    print(f"コンテンツ: {content}, 類似度: {similarity}")
            
            except Exception as e2:
                print(f"方法2でもエラーが発生しました: {e2}")
                
                # 方法3: RawクエリでSQLを直接構築
                try:
                    raw_embedding = ','.join(str(x) for x in embedding_array)
                    raw_sql = f"""
                    SELECT content, 1 - (embedding <=> '[{raw_embedding}]'::vector) AS similarity 
                    FROM items 
                    ORDER BY embedding <=> '[{raw_embedding}]'::vector 
                    LIMIT 3
                    """
                    cur.execute(raw_sql)
                    results = cur.fetchall()
                    
                    print(f"「{query}」に類似したアイテム (方法3):")
                    for content, similarity in results:
                        print(f"コンテンツ: {content}, 類似度: {similarity}")
                
                except Exception as e3:
                    print(f"方法3でもエラーが発生しました: {e3}")