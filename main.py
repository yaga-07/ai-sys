from src.extractors.pdf_extractor import PDFExtractor
from src.chunkers.simple_chunker import SimpleChunker
from src.llm_engines.google_engine import GoogleLLMEngine
from src.vector_stores.elasticsearch_store import ElasticsearchVectorStore
import os

def embed_texts(llm, texts, embedding_dim):
    # Use Gemini to embed texts (simulate with LLM for demo; replace with actual embedding API if available)
    vectors = []
    for text in texts:
        # For demonstration, use the LLM to generate a fake embedding (replace with real embedding logic)
        # Here, we just use the hash of the text mod 1.0 as a dummy vector
        import numpy as np
        np.random.seed(abs(hash(text)) % (2**32))
        vectors.append(np.random.rand(embedding_dim).tolist())
    return vectors

def run():
    # --- Setup ---
    pdf_path = "data/my_doc.pdf"
    embedding_dim = 768
    es_host = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
    index_name = "docs"

    extractor = PDFExtractor()
    chunker = SimpleChunker()
    llm = GoogleLLMEngine(model="gemini-1.5-pro")
    vector_store = ElasticsearchVectorStore(index_name=index_name, embedding_dim=embedding_dim, es_host=es_host)

    # --- Ingest ---
    text = extractor.extract(pdf_path)
    chunks = chunker.chunk(text)
    vectors = embed_texts(llm, chunks, embedding_dim)
    docs = [{"text": chunk, "vector": vector} for chunk, vector in zip(chunks, vectors)]
    vector_store.add_documents(docs)

    # --- Query ---
    user_query = "What is the document about?"
    query_vector = embed_texts(llm, [user_query], embedding_dim)[0]
    retrieved = vector_store.similarity_search(query_vector, k=3)
    context = "\n".join([doc["text"] for doc in retrieved])

    # --- Generate Answer ---
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    answer = llm.generate(messages)
    print(answer)

if __name__ == "__main__":
    run()
