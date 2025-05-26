from src.core.vector_store import BaseVectorStore
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import uuid
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ElasticsearchVectorStore(BaseVectorStore):
    def __init__(
        self,
        index_name: str,
        embedding_dim: int = 768,
        es_host: str = "http://localhost:9200",
        extra_mappings: dict = None,
    ):
        self.embedding_dim = embedding_dim
        self.index_name = index_name
        self.client = Elasticsearch(es_host)

        logger.info(f"Connecting to Elasticsearch at {es_host}")
        if not self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' does not exist. Creating index.")
            self._create_index(extra_mappings or {})
        else:
            logger.info(f"Index '{self.index_name}' already exists.")

    def _create_index(self, extra_mappings: dict):
        """
        Creates an index with vector, text, and any additional mapping fields.
        """
        mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    **extra_mappings
                }
            }
        }
        logger.debug(f"Creating index '{self.index_name}' with mapping: {mapping}")
        self.client.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Index '{self.index_name}' created successfully.")

    def add_documents(self, documents: list[dict]):
        """
        Add documents with structure:
        {
            "text": "...",
            "vector": [...],  # required, embedding vector
            "metadata": { arbitrary key/values }
        }
        """
        logger.info(f"Adding {len(documents)} documents to index '{self.index_name}'")
        for doc in documents:
            if "text" not in doc or "vector" not in doc:
                logger.error("Each document must contain 'text' and 'vector' fields")
                raise ValueError("Each document must contain 'text' and 'vector' fields")
            if not isinstance(doc["vector"], list) or len(doc["vector"]) != self.embedding_dim:
                logger.error(f"Each 'vector' must be a list of length {self.embedding_dim}")
                raise ValueError(f"Each 'vector' must be a list of length {self.embedding_dim}")

        actions = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            source = {
                "text": doc["text"],
                "vector": doc["vector"],
                **metadata  # flatten metadata fields
            }
            actions.append({
                "_index": self.index_name,
                "_id": str(uuid.uuid4()),
                "_source": source
            })

        logger.debug(f"Bulk indexing {len(actions)} documents")
        bulk(self.client, actions)
        logger.info(f"Successfully added {len(actions)} documents to index '{self.index_name}'")

    def similarity_search(self, query_vector: list[float], k: int, filters: dict = None):
        """
        Perform similarity search with optional filters on metadata fields.
        query_vector: list of floats, must match embedding_dim
        """
        logger.info(f"Performing similarity search on index '{self.index_name}' with k={k} and filters={filters}")
        if not isinstance(query_vector, list) or len(query_vector) != self.embedding_dim:
            logger.error(f"'query_vector' must be a list of length {self.embedding_dim}")
            raise ValueError(f"'query_vector' must be a list of length {self.embedding_dim}")

        base_query = {"match_all": {}}
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {key: value}})
                else:
                    filter_clauses.append({"term": {key: value}})
            base_query = {"bool": {"filter": filter_clauses}}

        search_query = {
            "size": k,
            "query": {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        logger.debug(f"Search query: {search_query}")
        response = self.client.search(index=self.index_name, body=search_query)
        logger.info(f"Similarity search returned {len(response['hits']['hits'])} results")
        return [hit["_source"] for hit in response["hits"]["hits"]]
