from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)  # hoặc Qdrant cloud

def search_similar_chunks(query_vector: list[float], top_k: int = 5):
    search_result = client.search(
        collection_name="my_documents",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        indexed_only=True  # Thêm parameter này
    )
    return search_result
