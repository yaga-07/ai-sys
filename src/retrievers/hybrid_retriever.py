from core.retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str) -> list[str]:
        # Implement hybrid retrieval logic here
        return self.vector_store.similarity_search(query, k=5)
