class BaseChunker:
    def chunk(self, text: str) -> list[str]:
        raise NotImplementedError
