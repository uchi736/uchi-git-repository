import psycopg2

print("PostgreSQL接続テスト開始...")
try:
    conn = psycopg2.connect(
        "dbname=postgres user=postgres password=password123 host=localhost port=5432"
    )
    print("接続成功！")

    cur = conn.cursor()
    cur.execute("SELECT version()")
    version = cur.fetchone()
    print(f"PostgreSQL バージョン: {version[0]}")

    # テーブル確認
    cur.execute("SELECT COUNT(*) FROM items")
    count = cur.fetchone()[0]
    print(f"itemsテーブルのレコード数: {count}")

    cur.close()
    conn.close()
    print("接続を閉じました")

except Exception as e:
    print(f"エラー: {type(e).__name__}: {e}")

print("テスト終了")
