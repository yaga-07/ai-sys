class BaseRetriever:
    def retrieve(self, query: str) -> list[str]:
        raise NotImplementedError
