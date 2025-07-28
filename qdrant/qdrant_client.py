from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "my_documents"

# Khởi tạo embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Kết nối Qdrant thật (chạy bằng Docker)
qdrant_client = QdrantClient(host="localhost", port=6333)

# Kiểm tra và tạo collection nếu chưa có
collections = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in collections:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def init_qdrant(collection_name="my_documents"):
    client = QdrantClient(host="localhost", port=6333)
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    return client

def search_vectors(query_text, top_k=5):
    query_embedding = embedder.encode(query_text).tolist()

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True  # Để trả lại thông tin nội dung
    )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in search_result
    ]
