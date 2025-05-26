from src.core.chunker import BaseChunker

class SlidingWindowChunker(BaseChunker):
    def chunk(self, text: str) -> list[str]:
        # Implement sliding window chunking logic here
        return [text]
