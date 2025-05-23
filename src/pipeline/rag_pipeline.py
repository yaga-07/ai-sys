class RAGPipeline:
    def __init__(self,
                 extractor,
                 chunker,
                 vector_store,
                 retriever,
                 llm_engine):
        self.extractor = extractor
        self.chunker = chunker
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm = llm_engine

    def ingest(self, file_path: str):
        text = self.extractor.extract(file_path)
        chunks = self.chunker.chunk(text)
        self.vector_store.add_documents(chunks)

    def query(self, query: str) -> str:
        retrieved_chunks = self.retriever.retrieve(query)
        context = "\n".join(retrieved_chunks)
        return self.llm.generate(f"Answer based on context:\n{context}\n\nQ: {query}\nA:")
