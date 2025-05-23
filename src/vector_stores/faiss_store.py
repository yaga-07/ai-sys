from core.vector_store import BaseVectorStore

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, embedding_model=None):
        self.docs = []

    def add_documents(self, docs):
        self.docs.extend(docs)

    def similarity_search(self, query, k):
        # Dummy: return first k docs
        return self.docs[:k]
