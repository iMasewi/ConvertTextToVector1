from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "my_documents"  # Đặt tên đúng với tên collection đã dùng

# Khởi tạo embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Khởi tạo Qdrant client
qdrant_client = QdrantClient(":memory:")  # Hoặc URL nếu dùng Qdrant Server

collections = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in collections:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def init_qdrant(collection_name="my_documents"):
    client = QdrantClient(":memory:")  # hoặc Qdrant URL nếu chạy server
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    return client
def search_vectors(query_text, top_k=5):
    # Convert query to embedding
    query_embedding = embedder.encode(query_text).tolist()

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in search_result
    ]