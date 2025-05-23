from config.factory import create_llm_engine
from retrievers.basic_retriever import BasicRetriever
from vector_stores.faiss_store import FAISSVectorStore
from extractors.pdf_extractor import PDFExtractor
from chunkers.simple_chunker import SimpleChunker
from pipeline.rag_pipeline import RAGPipeline

def run():
    extractor = PDFExtractor()
    chunker = SimpleChunker()
    vector_store = FAISSVectorStore()
    retriever = BasicRetriever(vector_store)
    llm = create_llm_engine("openai")

    pipeline = RAGPipeline(extractor, chunker, vector_store, retriever, llm)

    pipeline.ingest("data/my_doc.pdf")
    answer = pipeline.query("What is the document about?")
    print(answer)

if __name__ == "__main__":
    run()
