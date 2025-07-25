from uuid import uuid4
from qdrant_client.http.models import PointStruct

def upload_to_qdrant(client, collection_name, chunks, vectors):
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vec.tolist(),
            payload={"text": chunk}
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
