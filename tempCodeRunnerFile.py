    try:
            # ───────── 新旧 API 両対応 ─────────
            if hasattr(PGVector, "from_connection_string"):
                # langchain-postgres 系 or 古い community
                self.vector_store = PGVector.from_connection_string(
                    connection_string=self.connection_string,
                    embeddings=self.embeddings,
                    collection_name=self.config.collection_name,
                    use_jsonb=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )
            else:
                # 0.3 系 community はこちら
                self.vector_store = PGVector(
                    connection_string=self.connection_string,
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,   # ← 引数名に注意
                    use_jsonb=True,
                    distance_strategy=DistanceStrategy.COSINE,
                )

            print("PGVector ready.")

            self.retriever = HybridRetriever(
                self.vector_store, self.connection_string, self.config
            )

            prompt_template = ChatPromptTemplate.from_template(
                "あなたは親切で知識豊富なアシスタントです。以下のコンテキストを使用して質問に答えてください。\n"
                "回答は正確で、コンテキストに基づいている必要があります。\n"
                "コンテキストに答えが見つからない場合は「提供された情報からは回答できません」と答えてください。\n\n"
                "コンテキスト:\n{context}\n\n質問: {question}\n\n回答:"
            )

            self.chain = (
                {
                    "context": itemgetter("question") | self.retriever | self._format_docs,
                    "question": itemgetter("question"),
                }
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            print("RAG chain setup complete.")

        except Exception as e:
            print(f"PGVector 初期化失敗: {e}")
            raise