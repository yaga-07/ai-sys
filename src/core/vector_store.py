class BaseVectorStore:
    def add_documents(self, docs: list[str]):
        raise NotImplementedError

    def similarity_search(self, query: str, k: int):
        raise NotImplementedError
