from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Có thể thay bằng mô hình khác

def embed_chunks(chunks):
    vectors = model.encode(chunks, convert_to_numpy=True)
    return vectors

def embed_text(text: str) -> list[float]:
    return model.encode(text).tolist()
