import os
from sqlalchemy import create_engine, text
import sys # sys.exit() のために追加

# rag_system.py から Config クラスをインポートすることを想定
# もし rag_system.py が直接インポートできない場合は、
# Config クラスの定義をこのスクリプトにコピーするか、
# データベース接続情報を直接設定してください。
try:
    from rag_system import Config
except ImportError:
    print("エラー: rag_system.py から Config クラスをインポートできませんでした。")
    print("このスクリプトを rag_system.py と同じディレクトリに配置するか、")
    print("Configクラスの定義をこのスクリプト内にコピーしてください。")
    # Configクラスの簡易的な代替 (環境に合わせて実際の値を設定してください)
    class Config:
        db_host: str = os.getenv("DB_HOST", "localhost")
        db_port: str = os.getenv("DB_PORT", "5432")
        db_name: str = os.getenv("DB_NAME", "postgres")
        db_user: str = os.getenv("DB_USER", "postgres")
        db_password: str = os.getenv("DB_PASSWORD", "your-password") # 実際のパスワードを設定
    # もし上記のようにConfigをここで定義する場合、pg_dialectの決定ロジックも必要です。
    # print("スクリプト内の簡易Config定義を使用します。DB接続情報が正しいか確認してください。")
    # このままでは get_db_connection_string 内の pg_dialect 決定が不完全になる可能性があります。

def get_db_connection_string():
    """データベース接続文字列を生成します。"""
    cfg = Config() # Configクラスのインスタンスを作成
    
    pg_dialect = None
    try:
        import psycopg
        pg_dialect = "psycopg"
        # print("psycopg (v3) ドライバーを検出しました。")
    except ModuleNotFoundError:
        try:
            import psycopg2
            pg_dialect = "psycopg2"
            # print("psycopg2 ドライバーを検出しました。")
        except ModuleNotFoundError:
            print("エラー: psycopg または psycopg2 ドライバーが見つかりません。インストールしてください。")
            return None
            
    return f"postgresql+{pg_dialect}://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"

def clear_rag_tables(engine):
    """RAGシステム関連のテーブルから全データを削除します。"""
    # 削除するテーブルのリスト。依存関係を考慮し、通常はembeddingから先に削除します。
    tables_to_clear = [
        "langchain_pg_embedding", # PGVector の embedding データ
        "langchain_pg_collection",  # PGVector の collection 定義
        "document_chunks"         # カスタムテーブル
    ]
    
    with engine.connect() as connection:
        # トランザクション内で全ての削除処理を実行
        with connection.begin(): 
            for table_name in tables_to_clear:
                try:
                    print(f"テーブル「{table_name}」から全データを削除しています...")
                    # TRUNCATE TABLE の方が高速ですが、DELETE FROM の方が標準的で安全な場合があります。
                    # ここでは DELETE FROM を使用します。
                    stmt = text(f"DELETE FROM {table_name};")
                    result = connection.execute(stmt)
                    print(f"テーブル「{table_name}」から {result.rowcount} 行削除しました。")
                except Exception as e:
                    # テーブルが存在しない場合などのエラーを考慮
                    if "does not exist" in str(e).lower():
                         print(f"テーブル「{table_name}」が存在しないかアクセスできません。スキップします。")
                    else:
                        print(f"テーブル「{table_name}」のデータ削除中にエラーが発生しました: {e}")
                        print("エラーが発生したため、全ての操作をロールバックします。")
                        raise # エラーを再発生させてトランザクションをロールバック
            print("指定された全テーブルのデータ削除コマンドを実行しました（トランザクション内）。")
        # ここでトランザクションがコミットされます（エラーが発生しなかった場合）
    print("データ削除処理が完了しました。")

if __name__ == "__main__":
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 警告 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("このスクリプトは、データベース内の以下のテーブルからすべてのデータを完全に削除します。")
    print("  - document_chunks")
    print("  - langchain_pg_embedding")
    print("  - langchain_pg_collection")
    print("この操作は元に戻すことができず、恒久的なデータ損失につながります。")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    # ユーザーに最終確認を求める
    confirmation = input("本当にすべてのデータを削除してもよろしいですか？ 'yes' と入力して確認してください: ")
    
    if confirmation.lower() != 'yes':
        print("操作はキャンセルされました。")
        sys.exit() # スクリプトを終了

    print("\nデータ削除処理を開始します...")
    connection_string = get_db_connection_string()
    
    if not connection_string:
        print("データベース接続文字列の取得に失敗したため、処理を終了します。")
        sys.exit()

    try:
        # Configクラスからパスワードを取得してマスク処理（表示用）
        # この部分はConfigクラスが正しくロードされている場合にのみ機能します
        cfg_instance_for_masking = Config()
        masked_conn_string = connection_string.replace(cfg_instance_for_masking.db_password, "********")
    except Exception:
        masked_conn_string = connection_string # マスク失敗時はそのまま表示（ただし注意）
        print("警告: 接続文字列内のパスワードのマスクに失敗しました。")
        
    print(f"接続先データベース: {masked_conn_string}")
    
    try:
        engine = create_engine(connection_string)
        clear_rag_tables(engine)
        print("\n指定されたRAG関連テーブルのデータがすべて削除されたはずです。")
    except Exception as e:
        print(f"データベース操作中に予期せぬエラーが発生しました: {e}")