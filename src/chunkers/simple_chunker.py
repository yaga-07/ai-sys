from src.core.chunker import BaseChunker

class SimpleChunker(BaseChunker):
    def chunk(self, text: str) -> list[str]:
        # Simple split by paragraphs
        return text.split('\n\n')
