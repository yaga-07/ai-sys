class BaseVectorStore:
    def add_documents(self, docs: list):
        raise NotImplementedError

    def similarity_search(self, query: list[float], k: int):
        raise NotImplementedError
